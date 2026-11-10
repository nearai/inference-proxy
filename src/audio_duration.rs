//! Duration-only metering of an already uploaded file; no decoding, transcoding, or network I/O.
use std::io::{Cursor, ErrorKind};
use symphonia::core::{
    codecs::CODEC_TYPE_OPUS,
    errors::Error,
    formats::FormatOptions,
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
    probe::Hint,
    units::TimeBase,
};

pub(crate) fn seconds(bytes: Vec<u8>) -> Option<i32> {
    let source = MediaSourceStream::new(
        Box::new(Cursor::new(bytes)),
        MediaSourceStreamOptions::default(),
    );
    let mut format = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .ok()?
        .format;
    let track = format
        .tracks()
        .iter()
        .find(|track| track.codec_params.sample_rate.is_some())?;
    let id = track.id;
    let opus = track.codec_params.codec == CODEC_TYPE_OPUS;
    let time_base = if opus {
        TimeBase::new(1, 48_000)
    } else {
        track.codec_params.time_base?
    };
    let mut units = 0_u64;
    loop {
        match format.next_packet() {
            Ok(packet) if packet.track_id() == id => {
                // WebM often omits block durations. Opus stores the frame duration in its TOC.
                let duration = if opus {
                    opus_samples(&packet.data)?
                } else {
                    packet.dur
                };
                // Vorbis may emit an initial priming packet with zero duration.
                units = units.checked_add(duration)?;
            }
            Ok(_) => {}
            Err(Error::IoError(error)) if error.kind() == ErrorKind::UnexpectedEof => break,
            Err(_) => return None,
        }
    }
    // Count complete packets rather than n_frames, which can be a bitrate-based estimate.
    // Round once in integer arithmetic to avoid a floating point sum billing an extra second.
    if time_base.denom == 0 {
        return None;
    }
    let seconds =
        (u128::from(units) * u128::from(time_base.numer)).div_ceil(u128::from(time_base.denom));
    i32::try_from(seconds).ok().filter(|seconds| *seconds > 0)
}

// RFC 6716 sections 3.1 and 3.2: frame size and count are in the packet header.
fn opus_samples(packet: &[u8]) -> Option<u64> {
    let toc = *packet.first()?;
    let config = toc >> 3;
    let samples = match config {
        0..=11 => [480_u64, 960, 1920, 2880][usize::from(config & 3)],
        12..=15 => [480, 960][usize::from(config & 1)],
        _ => [120, 240, 480, 960][usize::from(config & 3)],
    };
    let count = match toc & 3 {
        0 => 1,
        1 | 2 => 2,
        _ => u64::from(*packet.get(1)? & 0x3f),
    };
    let total = samples * count;
    (count > 0 && total <= 5760).then_some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uploaded_formats_use_audio_frames_and_invalid_files_are_not_estimated_as_actual_usage() {
        for bytes in [
            include_bytes!("../tests/fixtures/audio/silence.wav").as_slice(),
            include_bytes!("../tests/fixtures/audio/silence.mp3").as_slice(),
            include_bytes!("../tests/fixtures/audio/silence.flac").as_slice(),
            include_bytes!("../tests/fixtures/audio/silence.m4a").as_slice(),
            include_bytes!("../tests/fixtures/audio/silence.ogg").as_slice(),
            include_bytes!("../tests/fixtures/audio/silence.webm").as_slice(),
        ] {
            assert_eq!(seconds(bytes.to_vec()), Some(2));
        }
        for bytes in [vec![], vec![0; 4096], b"not audio".to_vec()] {
            assert_eq!(seconds(bytes), None);
        }
    }

    #[test]
    fn opus_packet_duration_covers_frame_modes_and_rejects_invalid_counts() {
        for (packet, expected) in [
            (vec![0], Some(480)),
            (vec![3 << 3], Some(2880)),
            (vec![13 << 3], Some(960)),
            (vec![16 << 3], Some(120)),
            (vec![(19 << 3) | 1], Some(1920)),
            (vec![(19 << 3) | 3, 6], Some(5760)),
            (vec![(19 << 3) | 3, 7], None),
            (vec![3, 0], None),
            (vec![3], None),
        ] {
            assert_eq!(opus_samples(&packet), expected);
        }
    }
}
