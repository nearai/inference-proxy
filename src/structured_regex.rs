//! Static limits on the regexes behind structured outputs.
//!
//! A linear scan of the pattern text: no regex engine is built. A quantified
//! group made only of repeats is checked against the limits in `RegexLimits`;
//! a group with a required fixed atom in every branch, such as the `,` in
//! `([^,]{0,100},){0,100}`, is checked only for its single bounds.

/// Limits on one structured-output regex. `0` turns a limit off.
#[derive(Clone, Debug, PartialEq)]
pub struct RegexLimits {
    /// Outer bound times the largest variable-range bound inside it.
    pub max_nested_product: u64,
    /// Bound of a group that contains `*`, `+` or `{m,}`.
    pub max_star_repeat: u64,
    /// Any single `{m,n}` bound.
    pub max_bound: u64,
    /// Pattern length in characters.
    pub max_chars: usize,
}

impl Default for RegexLimits {
    fn default() -> Self {
        Self {
            max_nested_product: 64,
            max_star_repeat: 32,
            max_bound: 10_000,
            max_chars: 8192,
        }
    }
}

const MAX_DEPTH: usize = 64;
const INF: u64 = u64::MAX;

#[derive(Clone, Copy)]
struct Quant {
    min: u64,
    max: u64, // INF = unbounded
}

#[derive(Default, Clone, Copy)]
struct Inner {
    var: u64, // largest variable-range upper bound (or product) inside
    unbounded: bool,
    anchored: bool, // every branch has a required fixed atom
}

struct Scan<'a> {
    b: &'a [u8],
    i: usize,
    limits: &'a RegexLimits,
}

impl Scan<'_> {
    fn quant(&mut self) -> Result<Option<Quant>, String> {
        let Some(&c) = self.b.get(self.i) else {
            return Ok(None);
        };
        let q = match c {
            b'*' => Quant { min: 0, max: INF },
            b'+' => Quant { min: 1, max: INF },
            b'?' => Quant { min: 0, max: 1 },
            b'{' => match self.braces() {
                Some((q, end)) => {
                    self.i = end;
                    self.lazy();
                    return Ok(Some(q));
                }
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        self.i += 1;
        self.lazy();
        Ok(Some(q))
    }

    fn lazy(&mut self) {
        if matches!(self.b.get(self.i), Some(b'?') | Some(b'+')) {
            self.i += 1;
        }
    }

    /// `{m}`, `{m,}`, `{m,n}` or `{,n}` at `self.i`; `None` = a literal `{`.
    fn braces(&self) -> Option<(Quant, usize)> {
        let rest = &self.b[self.i + 1..];
        let close = rest.iter().position(|&c| c == b'}')?;
        let body = std::str::from_utf8(&rest[..close]).ok()?;
        let num = |s: &str| -> Option<u64> {
            if s.is_empty() || !s.bytes().all(|c| c.is_ascii_digit()) {
                return None;
            }
            Some(s.parse::<u64>().unwrap_or(INF))
        };
        let q = match body.split_once(',') {
            None => {
                let m = num(body)?;
                Quant { min: m, max: m }
            }
            Some((a, b)) => {
                let min = if a.is_empty() { 0 } else { num(a)? };
                let max = if b.is_empty() {
                    if a.is_empty() {
                        return None;
                    }
                    INF
                } else {
                    num(b)?
                };
                Quant { min, max }
            }
        };
        Some((q, self.i + 1 + close + 1))
    }

    fn check_bound(&self, q: Quant) -> Result<(), String> {
        let l = self.limits.max_bound;
        let top = if q.max == INF { q.min } else { q.max };
        if l > 0 && top > l {
            return Err(format!("a repeat bound is above {l}"));
        }
        Ok(())
    }

    /// One quantified item with `inner` contents: check it and return what it
    /// adds to the enclosing sequence.
    fn item(&self, inner: Inner, q: Option<Quant>) -> Result<Inner, String> {
        let Some(q) = q else {
            return Ok(inner);
        };
        self.check_bound(q)?;
        let upper = q.max;
        let lim = self.limits;
        if inner.anchored {
            return Ok(Inner {
                var: 0,
                unbounded: false,
                anchored: q.min >= 1,
            });
        }
        if inner.var > 0 && upper > 1 {
            let product = if upper == INF {
                INF
            } else {
                upper.saturating_mul(inner.var)
            };
            if lim.max_nested_product > 0 && product > lim.max_nested_product {
                return Err(
                    "a variable-length repeat is nested inside a repeat that is too large"
                        .to_string(),
                );
            }
        }
        if inner.unbounded && upper != INF && lim.max_star_repeat > 0 && upper > lim.max_star_repeat
        {
            return Err(format!(
                "a group containing `*` or `+` is repeated more than {} times",
                lim.max_star_repeat
            ));
        }
        let variable = q.max != q.min;
        let var = if variable && upper != INF {
            inner.var.max(upper.saturating_mul(inner.var.max(1)))
        } else {
            inner.var
        };
        Ok(Inner {
            var,
            unbounded: inner.unbounded || upper == INF,
            anchored: false,
        })
    }

    fn seq(&mut self, depth: usize) -> Result<Inner, String> {
        if depth > MAX_DEPTH {
            return Err("groups are nested too deeply".to_string());
        }
        let mut acc = Inner::default();
        let mut all_anchored = true;
        let mut branch_anchored = false;
        while let Some(&c) = self.b.get(self.i) {
            let inner = match c {
                b')' => break,
                b'|' => {
                    all_anchored &= branch_anchored;
                    branch_anchored = false;
                    self.i += 1;
                    continue;
                }
                b'^' | b'$' => {
                    self.i += 1;
                    continue;
                }
                b'(' => {
                    self.i += 1;
                    if self.b.get(self.i) == Some(&b'?') {
                        // (?:  (?P<name>  (?<name>  (?=  (?!  (?i) ...
                        while let Some(&c) = self.b.get(self.i) {
                            self.i += 1;
                            if matches!(c, b':' | b'>' | b'=' | b'!') {
                                break;
                            }
                            if c == b')' {
                                self.i -= 1;
                                break;
                            }
                        }
                    }
                    let inner = self.seq(depth + 1)?;
                    if self.b.get(self.i) == Some(&b')') {
                        self.i += 1;
                    }
                    inner
                }
                b'[' => {
                    self.i += 1;
                    if self.b.get(self.i) == Some(&b'^') {
                        self.i += 1;
                    }
                    if self.b.get(self.i) == Some(&b']') {
                        self.i += 1;
                    }
                    while let Some(&c) = self.b.get(self.i) {
                        self.i += if c == b'\\' { 2 } else { 1 };
                        if c == b']' {
                            break;
                        }
                    }
                    Inner::default()
                }
                b'\\' => {
                    self.i += 2;
                    Inner::default()
                }
                _ => {
                    self.i += 1;
                    Inner::default()
                }
            };
            let q = self.quant()?;
            let fixed_atom = inner.var == 0 && !inner.unbounded;
            let it = self.item(inner, q)?;
            let required = match q {
                None => true,
                Some(q) => q.min >= 1 && q.min == q.max,
            };
            branch_anchored |= it.anchored || (fixed_atom && required);
            acc.var = acc.var.max(it.var);
            acc.unbounded |= it.unbounded;
        }
        acc.anchored = all_anchored && branch_anchored;
        Ok(acc)
    }
}

/// `Err(reason)` when `pattern` is outside `limits`.
pub fn check_regex(pattern: &str, limits: &RegexLimits) -> Result<(), String> {
    if limits.max_chars > 0 && pattern.chars().count() > limits.max_chars {
        return Err(format!(
            "the pattern is longer than {} characters",
            limits.max_chars
        ));
    }
    let mut scan = Scan {
        b: pattern.as_bytes(),
        i: 0,
        limits,
    };
    while scan.i < scan.b.len() {
        scan.seq(0)?;
        // A stray `)` at top level: skip it and go on.
        scan.i += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slow_patterns_are_refused() {
        let l = RegexLimits::default();
        for p in [
            "(.{0,10}){0,10}",
            "(.{0,20}){0,20}",
            "^(.{0,500}){0,500}$",
            "([a-z]{0,50}){0,50}",
            "(.{1,3}){0,100}",
            "(.{0,10})*",
            "(.{0,10})+",
            "(.*){0,50}",
            "^a{1000000}$",
            "(?:(?:.{0,9}){0,9}){0,9}",
            "(x|.{0,50}){0,50}",
            "(x?.{0,50}){0,50}",
            "(^.{0,50}$){0,50}",
        ] {
            assert!(check_regex(p, &l).is_err(), "{p}");
        }
    }

    #[test]
    fn ordinary_patterns_pass() {
        let l = RegexLimits::default();
        for p in [
            r"\d{4}-\d{2}-\d{2}",
            "[a-z]{1,64}",
            r"(\d{3}-){0,10}\d{4}",
            r"(\w+\s)*",
            r"(\w+\s){0,16}",
            "(.*)*",
            "(.*){0,5}",
            "(a|b){0,1000}",
            "(.{0,3}){0,3}",
            "(.{0,5}){0,5}",
            "[A-Z][a-z]{1,20}( [A-Z][a-z]{1,20}){0,3}",
            "(.{3}){0,1000}",
            "a{100}",
            r"\{not a quantifier}",
            r"(\w{1,20} ){0,500}",
            "([^,]{0,100},){0,100}",
            "(.{0,50}x){0,50}",
            "([A-Za-z0-9._-]{1,64}/){0,8}[A-Za-z0-9._-]{1,64}",
            "(.{1,20}\n){0,10}",
            r"(\d{1,3}\.){3}\d{1,3}",
            "[({]{0,5}",
            "(?P<year>[0-9]{4})-(?:[0-9]{2})",
            "^[^@]+@[^@]+\\.[a-z]{2,10}$",
        ] {
            assert!(check_regex(p, &l).is_ok(), "{p}");
        }
    }

    #[test]
    fn limits_can_be_turned_off_and_long_patterns_are_refused() {
        let off = RegexLimits {
            max_nested_product: 0,
            max_star_repeat: 0,
            max_bound: 0,
            max_chars: 0,
        };
        assert!(check_regex("(.{0,500}){0,500}", &off).is_ok());
        assert!(check_regex(&"a".repeat(9000), &RegexLimits::default()).is_err());
    }

    #[test]
    fn scan_is_fast() {
        let p = "[a-z]{1,20}(-[a-z0-9]{1,8}){0,2}".repeat(60);
        let t = std::time::Instant::now();
        for _ in 0..100 {
            let _ = check_regex(&p, &RegexLimits::default());
        }
        assert!(t.elapsed().as_millis() < 100);
    }
}
