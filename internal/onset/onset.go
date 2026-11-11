// Package onset finds where speech actually starts in the audio.
//
// whisper's segment start times often fall in the silence before a line (right where the previous
// line ended), sometimes seconds early. Given whisper's rough start, Refine looks at the voice-band
// energy of the real audio and returns the moment speech rises out of quiet.
package onset

import (
	"encoding/binary"
	"math"
	"sort"
)

const (
	sampleRate = 16000
	frameLen   = 160 // 10 ms
	frameSec   = float64(frameLen) / sampleRate
)

// Detector holds the voice-band loudness (dB) of every 10 ms frame of one audio chunk.
type Detector struct {
	db []float64
}

type biquad struct{ b0, b1, b2, a1, a2, x1, x2, y1, y2 float64 }

func (q *biquad) step(x float64) float64 {
	y := q.b0*x + q.b1*q.x1 + q.b2*q.x2 - q.a1*q.y1 - q.a2*q.y2
	q.x2, q.x1, q.y2, q.y1 = q.x1, x, q.y1, y
	return y
}

// RBJ cookbook filters, Q = 0.707
func lowpass(f float64) *biquad {
	w := 2 * math.Pi * f / sampleRate
	al := math.Sin(w) / (2 * 0.7071)
	c := math.Cos(w)
	a0 := 1 + al
	return &biquad{b0: (1 - c) / 2 / a0, b1: (1 - c) / a0, b2: (1 - c) / 2 / a0, a1: -2 * c / a0, a2: (1 - al) / a0}
}

func highpass(f float64) *biquad {
	w := 2 * math.Pi * f / sampleRate
	al := math.Sin(w) / (2 * 0.7071)
	c := math.Cos(w)
	a0 := 1 + al
	return &biquad{b0: (1 + c) / 2 / a0, b1: -(1 + c) / a0, b2: (1 + c) / 2 / a0, a1: -2 * c / a0, a2: (1 - al) / a0}
}

// New analyses 16 kHz mono s16le PCM.
func New(pcm []byte) *Detector {
	n := len(pcm) / 2
	hp1, hp2 := highpass(300), highpass(300)
	lp1, lp2 := lowpass(3400), lowpass(3400)
	d := &Detector{db: make([]float64, 0, n/frameLen+1)}
	var acc float64
	k := 0
	for i := 0; i < n; i++ {
		x := float64(int16(binary.LittleEndian.Uint16(pcm[2*i:]))) / 32768
		y := lp2.step(lp1.step(hp2.step(hp1.step(x))))
		acc += y * y
		if k++; k == frameLen {
			d.db = append(d.db, 10*math.Log10(acc/frameLen+1e-12))
			acc, k = 0, 0
		}
	}
	return d
}

func (d *Detector) frame(t float64) int {
	f := int(math.Round(t / frameSec))
	if f < 0 {
		return 0
	}
	if f > len(d.db) {
		return len(d.db)
	}
	return f
}

func percentile(xs []float64, p float64) float64 {
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	return s[int(p*float64(len(s)-1))]
}

// Refine returns where speech really starts for a segment whisper says starts at t, searching
// from just before t up to `until` (the next segment's start). ok is false when the audio shows
// no clear onset there: continuous speech or loud background, or t already inside running speech
// (then any later onset would belong to a different phrase).
func (d *Detector) Refine(t, until float64) (float64, bool) {
	if until <= t {
		until = t + 4
	}
	until = math.Min(until, t+15) // whisper's start can sit at the previous line's end, long before
	lo, hi := d.frame(t-0.3), d.frame(until)
	ctx := d.db[d.frame(t-1.5):hi]
	if hi-lo < 14 || len(ctx) < 20 {
		return t, false
	}
	floor, peak := percentile(ctx, 0.15), percentile(ctx, 0.95)
	if peak-floor < 8 { // no real contrast between speech and background
		return t, false
	}
	thr := floor + 0.35*(peak-floor)
	attack := floor + 0.15*(peak-floor)

	// speech-like: 120 ms mostly above the threshold (a click, cough or music hit is shorter)
	sustained := func(f int) bool {
		above := 0
		for k := f; k < f+12; k++ {
			if d.db[k] >= thr {
				above++
			}
		}
		return d.db[f] >= thr && above >= 10
	}
	// ... coming out of mostly quiet (the previous 150 ms)
	fromQuiet := func(f int) bool {
		quiet, n := 0, 0
		for k := f - 15; k < f; k++ {
			if k < 0 {
				continue
			}
			n++
			if d.db[k] < thr {
				quiet++
			}
		}
		return n >= 10 && float64(quiet) >= 0.8*float64(n)
	}
	ft := d.frame(t)
	inSpeech := ft+12 < len(d.db) && sustained(ft)

	for f := lo; f+12 < hi; f++ {
		if !sustained(f) || !fromQuiet(f) {
			continue
		}
		if inSpeech && float64(f)*frameSec > t+0.1 {
			// t is inside speech that started before our search: the onset we'd report is the
			// next phrase, not this one
			return t, false
		}
		// back up to where the attack began - at most 100 ms, never through a gap into the
		// previous line
		for k := 0; k < 10 && f > 0 && d.db[f-1] >= attack; k++ {
			f--
		}
		if f > 0 && d.db[f-1] >= attack {
			return t, false // still rising after 100 ms: not a clean onset
		}
		return float64(f) * frameSec, true
	}
	return t, false
}
