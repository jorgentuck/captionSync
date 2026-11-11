package align

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// Realistic conditions seen on real films: only a fraction of cues get a trustworthy anchor
// (start of a whisper segment), timestamps carry ~200 ms of noise, and some matches are wrong.
type scene struct {
	cues      int
	yield     float64 // fraction of cues whose first word is a reliable segment start
	noise     float64 // std-dev of reliable timestamps, seconds
	outliers  float64 // fraction of reliable anchors that are simply wrong
	speechOK  func(t float64) bool
	subOf     func(audio float64) float64 // where the subtitle file (wrongly) puts a line spoken at `audio`
	seed      int64
	truth     []float64 // filled in: true speech start per cue
	subsTimes []float64 // filled in: subtitle start per cue
}

func (sc *scene) build() ([]Cue, []Word) {
	r := rand.New(rand.NewSource(sc.seed))
	var cues []Cue
	var words []Word
	t := 20.0
	for i := 0; i < sc.cues; i++ {
		k := 4 + r.Intn(5)
		toks := make([]string, k)
		for j := range toks {
			toks[j] = vocab[r.Intn(len(vocab))]
		}
		dur := 0.32 * float64(k)
		s := sc.subOf(t)
		cues = append(cues, Cue{Start: s, End: sc.subOf(t + dur + 0.4), Tokens: toks})
		sc.truth = append(sc.truth, t)
		sc.subsTimes = append(sc.subsTimes, s)
		if sc.speechOK == nil || sc.speechOK(t) {
			reliable := r.Float64() < sc.yield
			at := t + r.NormFloat64()*sc.noise
			if reliable && r.Float64() < sc.outliers {
				at = t + (r.Float64()*2-1)*30 // a confident but wrong match
			}
			if !reliable {
				at = t + 0.5 + r.Float64()*2.5 // mid-segment word: time smeared late, not trustworthy
			}
			for j, tk := range toks {
				words = append(words, Word{Text: tk, Start: at + float64(j)*0.3, Reliable: reliable && j == 0})
			}
		}
		t += dur + 1.5 + r.Float64()*6
	}
	return cues, words
}

// fraction of cues landing within tol seconds of the real speech after applying the result
func accuracy(sc *scene, r Result, tol float64) float64 {
	if r.Mapping == nil {
		return 0
	}
	ok := 0
	for i, s := range sc.subsTimes {
		got, _ := r.Mapping.Apply(s, s+1)
		if math.Abs(got-sc.truth[i]) <= tol {
			ok++
		}
	}
	return float64(ok) / float64(len(sc.subsTimes))
}

func run(sc *scene, sampled bool) Result {
	cues, words := sc.build()
	dur := sc.truth[len(sc.truth)-1] + 30
	sort.Slice(cues, func(i, j int) bool { return cues[i].Start < cues[j].Start })
	return Analyze(cues, words, nil, [][2]float64{{0, dur}}, dur, DefaultParams(), sampled)
}

// "Adventures in Babysitting": steady drift (~3.3 s/h) must be ONE sloped line, not six steps.
func TestRealisticDriftIsOneLine(t *testing.T) {
	sc := &scene{cues: 1100, yield: 0.13, noise: 0.2, outliers: 0.15, seed: 21,
		subOf: func(a float64) float64 { return a + 1.7 + a*3.3/3600 }}
	r := run(sc, false)
	if r.Status != "needs_fix" {
		t.Fatalf("status %s: %s", r.Status, r.Message)
	}
	if len(r.Mapping.Pieces) != 1 {
		t.Errorf("expected 1 piece, got %d: %s", len(r.Mapping.Pieces), r.Message)
	}
	if acc := accuracy(sc, r, 0.25); acc < 0.95 {
		t.Errorf("only %.0f%% of lines within 250 ms: %s", acc*100, r.Message)
	}
}

// "9 (2009)": a plain constant offset with noisy anchors must not grow a fake drift.
func TestRealisticNoFakeDrift(t *testing.T) {
	for seed := int64(1); seed <= 8; seed++ {
		sc := &scene{cues: 390, yield: 0.18, noise: 0.22, outliers: 0.2, seed: seed,
			subOf: func(a float64) float64 { return a - 0.6 }}
		r := run(sc, false)
		if r.Status != "needs_fix" {
			t.Fatalf("seed %d status %s: %s", seed, r.Status, r.Message)
		}
		if acc := accuracy(sc, r, 0.2); acc < 0.97 {
			t.Errorf("seed %d: only %.0f%% within 200 ms: %s", seed, acc*100, r.Message)
		}
	}
}

// "2001": very little dialogue in the sampled windows -> don't trust a handful of anchors;
// ask for a full pass instead of writing a guess.
func TestRealisticSparseSampleEscalates(t *testing.T) {
	windows := func(a float64) bool { // 12 one-minute windows over ~2.5 h, dialogue only in a few
		w := int(a / 750)
		return int(a)%750 < 60 && (w == 3 || w == 9)
	}
	sc := &scene{cues: 640, yield: 0.15, noise: 0.25, outliers: 0.2, seed: 5, speechOK: windows,
		subOf: func(a float64) float64 { return a - 1.4 }}
	r := run(sc, true)
	if !r.WantsFull {
		t.Errorf("expected escalation to a full pass, got %s: %s", r.Status, r.Message)
	}
	if r.Status == "needs_fix" && accuracy(sc, r, 0.25) < 0.95 {
		t.Errorf("wrote a weak fix: %s", r.Message)
	}
}

// Weak evidence even after a full pass -> leave the file alone.
func TestRealisticWeakFullPassIsUncertain(t *testing.T) {
	sc := &scene{cues: 300, yield: 0.03, noise: 0.3, outliers: 0.45, seed: 9,
		subOf: func(a float64) float64 { return a + 2 }}
	r := run(sc, false)
	if r.Status == "needs_fix" && accuracy(sc, r, 0.3) < 0.9 {
		t.Errorf("wrote an unreliable fix: %s", r.Message)
	}
}

// A genuine jump (different cut) still becomes two pieces, placed correctly.
func TestRealisticRealJump(t *testing.T) {
	sc := &scene{cues: 900, yield: 0.15, noise: 0.2, outliers: 0.15, seed: 13,
		subOf: func(a float64) float64 {
			if a > 2400 {
				return a + 3.2
			}
			return a + 0.9
		}}
	r := run(sc, false)
	if r.Status != "needs_fix" || len(r.Mapping.Pieces) != 2 {
		t.Fatalf("expected 2 pieces: %s", r.Message)
	}
	if acc := accuracy(sc, r, 0.25); acc < 0.95 {
		t.Errorf("only %.0f%% within 250 ms: %s", acc*100, r.Message)
	}
}

// Drift plus a jump: line fit per section.
func TestRealisticDriftAndJump(t *testing.T) {
	sc := &scene{cues: 1100, yield: 0.15, noise: 0.2, outliers: 0.15, seed: 17,
		subOf: func(a float64) float64 {
			s := a + a*3.6/3600
			if a > 3000 {
				s += 4
			}
			return s
		}}
	r := run(sc, false)
	if r.Status != "needs_fix" {
		t.Fatalf("status %s: %s", r.Status, r.Message)
	}
	if acc := accuracy(sc, r, 0.3); acc < 0.93 {
		t.Errorf("only %.0f%% within 300 ms: %s", acc*100, r.Message)
	}
}

// --- regressions from the multi-agent review: every result must be correct or uncertain ---------

// windowsOf builds 12 x 60 s sampled windows like the processor does.
func windowsOf(dur float64) [][2]float64 {
	m := math.Min(120, dur*0.03)
	u := dur - 2*m - 60
	var w [][2]float64
	for i := 0; i < 12; i++ {
		s := m + u*float64(i)/11
		w = append(w, [2]float64{s, s + 60})
	}
	return w
}

func inWindows(w [][2]float64) func(float64) bool {
	return func(t float64) bool {
		for _, x := range w {
			if t >= x[0] && t < x[1] {
				return true
			}
		}
		return false
	}
}

// never: a written fix (needs_fix) must put at least `need` of the cues in `region` within tol.
func checkRegion(t *testing.T, name string, sc *scene, r Result, region func(float64) bool, tol, need float64) bool {
	if r.Status != "needs_fix" && r.Status != "in_sync" {
		return true // uncertain (or escalation) is an acceptable answer
	}
	ok, n := 0, 0
	for i, s := range sc.subsTimes {
		if !region(sc.truth[i]) {
			continue
		}
		n++
		got := s
		if r.Mapping != nil {
			got, _ = r.Mapping.Apply(s, s+1)
		}
		if math.Abs(got-sc.truth[i]) <= tol {
			ok++
		}
	}
	if n > 0 && float64(ok)/float64(n) < need {
		t.Errorf("%s: %s %q left %d/%d region cues within %.0f ms", name, r.Status, r.Message, ok, n, tol*1000)
		return false
	}
	return true
}

func TestReviewColdOpen(t *testing.T) {
	for seed := int64(1); seed <= 20; seed++ {
		sc := &scene{cues: 420, yield: 0.3, noise: 0.2, outliers: 0.15, seed: seed,
			subOf: func(a float64) float64 {
				if a < 240 {
					return a - 1.5
				}
				return a + 1.0
			}}
		r := run(sc, false)
		checkRegion(t, "cold open", sc, r, func(a float64) bool { return a < 240 }, 0.3, 0.9)
		checkRegion(t, "body", sc, r, func(a float64) bool { return a >= 300 }, 0.3, 0.9)
	}
}

func TestReviewLateTail(t *testing.T) {
	for seed := int64(1); seed <= 20; seed++ {
		sc := &scene{cues: 1100, yield: 0.15, noise: 0.2, outliers: 0.15, seed: seed}
		var cut float64
		sc.subOf = func(a float64) float64 {
			if cut > 0 && a > cut {
				return a - 1.5
			}
			return a + 1.0
		}
		cut = 0.94 * (20 + 1100*5.5) // ~94% of the runtime
		r := run(sc, false)
		checkRegion(t, "tail", sc, r, func(a float64) bool { return a > cut+30 }, 0.3, 0.9)
		checkRegion(t, "body", sc, r, func(a float64) bool { return a < cut-30 }, 0.3, 0.9)
	}
}

func TestReviewSampledLateJumpEscalates(t *testing.T) {
	for seed := int64(1); seed <= 20; seed++ {
		sc := &scene{cues: 1300, yield: 0.15, noise: 0.2, outliers: 0.15, seed: seed}
		dur := 20 + 1300*5.5
		w := windowsOf(dur)
		sc.speechOK = inWindows(w)
		sc.subOf = func(a float64) float64 {
			if a > 0.75*dur {
				return a + 2.5
			}
			return a
		}
		cues, words := sc.build()
		sort.Slice(cues, func(i, j int) bool { return cues[i].Start < cues[j].Start })
		r := Analyze(cues, words, nil, w, dur, DefaultParams(), true)
		if r.WantsFull {
			continue // escalates to a full pass: correct behaviour
		}
		checkRegion(t, "sampled tail", sc, r, func(a float64) bool { return a > 0.75*dur+30 }, 0.3, 0.9)
		checkRegion(t, "sampled body", sc, r, func(a float64) bool { return a < 0.75*dur-30 }, 0.3, 0.9)
	}
}

func TestReviewExcursionIsNotAFix(t *testing.T) {
	for seed := int64(1); seed <= 20; seed++ {
		sc := &scene{cues: 900, yield: 0.2, noise: 0.2, outliers: 0.1, seed: seed,
			subOf: func(a float64) float64 {
				if a > 2000 && a < 2100 { // brief excursion that returns to the same offset
					return a + 3.0
				}
				return a + 0.8
			}}
		r := run(sc, false)
		checkRegion(t, "outside excursion", sc, r, func(a float64) bool { return a < 1950 || a > 2150 }, 0.3, 0.9)
	}
}

func TestReviewZeroMinAnchorsDoesNotPanic(t *testing.T) {
	p := DefaultParams()
	p.MinAnchors = 0
	_ = Analyze(nil, nil, nil, nil, 100, p, false)
	_ = Analyze([]Cue{{Start: 1, End: 2, Tokens: []string{"hello", "there", "friend"}}}, nil, nil, nil, 100, p, false)
	if m := FitMapping(nil, nil, nil, nil, p); m != nil {
		t.Fatal("mapping from nothing")
	}
}
