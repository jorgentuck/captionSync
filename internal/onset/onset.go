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

// levels returns the background floor and speech peak around a stretch of audio.
func (d *Detector) levels(from, to int) (floor, peak float64, ok bool) {
	if from < 0 {
		from = 0
	}
	if to > len(d.db) {
		to = len(d.db)
	}
	if to-from < 20 {
		return 0, 0, false
	}
	ctx := d.db[from:to]
	floor, peak = percentile(ctx, 0.15), percentile(ctx, 0.95)
	return floor, peak, peak-floor >= 8 // no real contrast between speech and background
}

// fraction of frames in [a, b) at or above level
func (d *Detector) above(a, b int, level float64) float64 {
	if a < 0 {
		a = 0
	}
	if b > len(d.db) {
		b = len(d.db)
	}
	if b <= a {
		return 0
	}
	n := 0
	for k := a; k < b; k++ {
		if d.db[k] >= level {
			n++
		}
	}
	return float64(n) / float64(b-a)
}

// ActiveAtStart reports whether the chunk already opens on sound loud enough to be speech
// (i.e. it may have been cut mid-sentence).
func (d *Detector) ActiveAtStart() bool {
	floor, peak, ok := d.levels(0, len(d.db))
	if !ok {
		return false
	}
	return d.above(0, 15, floor+0.25*(peak-floor)) >= 0.5
}

// Refine returns where speech really starts for a segment whisper says starts at t, searching
// from just before t up to `until` (the next segment's start). ok is false when the audio shows
// no clear, speech-like onset there - continuous speech or loud background, t inside running
// speech, or speech between t and the candidate (then the candidate belongs to a later phrase).
func (d *Detector) Refine(t, until float64) (float64, bool) {
	if until <= t {
		until = t + 4
	}
	until = math.Min(until, t+15) // whisper's start can sit at the previous line's end, long before
	lo, hi := d.frame(t-0.3), d.frame(until)
	if hi-lo < 14 {
		return t, false
	}
	floor, peak, ok := d.levels(d.frame(t-1.5), hi)
	if !ok {
		return t, false
	}
	thr := floor + 0.35*(peak-floor)
	attack := floor + 0.15*(peak-floor)
	ft := d.frame(t)

	// t is inside speech only if speech keeps going well after it (not just the previous line's tail)
	inSpeech := d.above(ft, ft+60, attack) >= 0.7

	for f := lo; f+12 < hi; f++ {
		// candidate: 120 ms mostly loud, coming out of mostly quiet (the previous 150 ms)
		if d.db[f] < thr || d.above(f, f+12, thr) < 0.83 || d.above(f-15, f, thr) > 0.2 || f < 10 {
			continue
		}
		far := float64(f)*frameSec > t+4
		// speech-like: followed by sustained energy (400 ms, or 600 ms when far from whisper's start),
		// so a door slam, laugh or music hit is not taken for the line
		need := 40
		if far {
			need = 60
		}
		if d.above(f, f+need, attack) < 0.75 {
			continue
		}
		// a short burst that falls back to quiet (door, laugh, music sting) loses to a later onset
		if d.burst(f, attack) && d.laterCandidate(f, hi, thr, attack) {
			continue
		}
		after := float64(f)*frameSec > t+0.1
		if after && (inSpeech || d.above(ft+10, f-15, attack) > 0.3) {
			// t sits in running speech, or there is speech between t and this onset: the onset
			// belongs to a later phrase, not to this segment
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

// burst: the loud run starting at f falls back to 200 ms of quiet within 600 ms.
func (d *Detector) burst(f int, attack float64) bool {
	for k := f + 10; k < f+60 && k+20 <= len(d.db); k++ {
		if d.above(k, k+20, attack) < 0.1 {
			return true
		}
	}
	return false
}

// laterCandidate: another sustained, speech-like onset follows f before hi.
func (d *Detector) laterCandidate(f, hi int, thr, attack float64) bool {
	for g := f + 30; g+40 < hi; g++ {
		if d.db[g] >= thr && d.above(g, g+12, thr) >= 0.83 && d.above(g-15, g, thr) <= 0.2 &&
			d.above(g, g+40, attack) >= 0.75 && !d.burst(g, attack) {
			return true
		}
	}
	return false
}
