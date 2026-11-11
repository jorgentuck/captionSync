// Package library knows which files are videos/subtitles, finds sidecars and computes change signatures.
package library

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/jorgentuck/captionSync/internal/config"
)

type Library struct{ cfg *config.Config }

func New(cfg *config.Config) *Library { return &Library{cfg: cfg} }

// MapPath translates a path as another container (e.g. Tdarr) sees it into ours (longest
// matching mapping wins).
func (l *Library) MapPath(p string) string {
	p = strings.TrimSpace(p)
	best := -1
	for i, m := range l.cfg.PathMappings {
		if (p == m[0] || strings.HasPrefix(p, m[0]+"/")) && (best < 0 || len(m[0]) > len(l.cfg.PathMappings[best][0])) {
			best = i
		}
	}
	if best < 0 {
		return p
	}
	m := l.cfg.PathMappings[best]
	return m[1] + p[len(m[0]):]
}

// SharedStem reports whether another video in the same folder has the same name (Movie.mkv and
// Movie.mp4): their subtitle files can't be attributed to one of them safely.
func (l *Library) SharedStem(video string) bool {
	entries, err := os.ReadDir(filepath.Dir(video))
	if err != nil {
		return false
	}
	me, st := filepath.Base(video), stem(filepath.Base(video))
	for _, e := range entries {
		if !e.IsDir() && e.Name() != me && l.IsVideo(e.Name()) && stem(e.Name()) == st {
			return true
		}
	}
	return false
}

// Allowed reports whether p lies inside one of the configured media roots.
// Symlinks are resolved first, so a link inside a root can't lead outside it.
func (l *Library) Allowed(p string) bool {
	p = filepath.Clean(p)
	if !within(p, l.cfg.MediaRoots) {
		return false
	}
	real := resolve(p)
	var roots []string
	for _, r := range l.cfg.MediaRoots {
		roots = append(roots, resolve(r))
	}
	return within(real, roots)
}

func within(p string, roots []string) bool {
	for _, r := range roots {
		if p == r || strings.HasPrefix(p, strings.TrimRight(r, "/")+"/") {
			return true
		}
	}
	return false
}

// resolve follows symlinks; for a path that doesn't exist yet it resolves the closest existing parent.
func resolve(p string) string {
	if r, err := filepath.EvalSymlinks(p); err == nil {
		return r
	}
	if parent := filepath.Dir(p); parent != p {
		return filepath.Join(resolve(parent), filepath.Base(p))
	}
	return p
}

// Ignored matches the ignore patterns against the file name and every directory component.
func (l *Library) Ignored(p string) bool {
	for _, part := range strings.Split(filepath.ToSlash(p), "/") {
		if part == "" {
			continue
		}
		for _, pat := range l.cfg.IgnorePatterns {
			if ok, _ := filepath.Match(pat, part); ok {
				return true
			}
		}
	}
	return false
}

func ext(p string) string { return strings.ToLower(filepath.Ext(p)) }

func has(list []string, s string) bool {
	for _, x := range list {
		if x == s {
			return true
		}
	}
	return false
}

func (l *Library) IsVideo(p string) bool {
	return has(l.cfg.VideoExt, ext(p)) && !l.Ignored(filepath.Base(p))
}

func (l *Library) IsSubtitle(p string) bool {
	return has(l.cfg.SubExt, ext(p)) && !l.Ignored(filepath.Base(p))
}

func stem(name string) string { return strings.TrimSuffix(name, filepath.Ext(name)) }

// owner returns the video stem a subtitle file belongs to (longest matching prefix wins),
// so "Show.S01E01.en.srt" belongs to "Show.S01E01" and not to "Show".
func owner(subName string, videoStems []string) string {
	s := stem(subName)
	best := ""
	for _, vs := range videoStems {
		if (s == vs || strings.HasPrefix(s, vs+".")) && len(vs) > len(best) {
			best = vs
		}
	}
	return best
}

// videoPriority orders videos that share a name (Movie.mkv, Movie.mp4): only the first one owns
// the shared subtitle files, so they are never retimed back and forth against two different videos.
func videoPriority(name string) int {
	switch ext(name) {
	case ".mkv":
		return 0
	case ".mp4":
		return 1
	case ".m4v":
		return 2
	}
	return 3
}

// videosByStem maps each video stem in dir to the one video that owns that stem's subtitles.
func (l *Library) videosByStem(entries []os.DirEntry) map[string]string {
	by := map[string]string{}
	for _, e := range entries {
		if e.IsDir() || !l.IsVideo(e.Name()) {
			continue
		}
		st := stem(e.Name())
		cur, ok := by[st]
		if !ok || videoPriority(e.Name()) < videoPriority(cur) ||
			(videoPriority(e.Name()) == videoPriority(cur) && e.Name() < cur) {
			by[st] = e.Name()
		}
	}
	return by
}

func (l *Library) Sidecars(video string) []string {
	dir := filepath.Dir(video)
	me := filepath.Base(video)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	by := l.videosByStem(entries)
	if o, ok := by[stem(me)]; ok && o != me {
		return nil // another video with the same name owns these subtitles
	}
	by[stem(me)] = me
	stems := make([]string, 0, len(by))
	for st := range by {
		stems = append(stems, st)
	}
	var out []string
	for _, e := range entries {
		if !e.IsDir() && l.IsSubtitle(e.Name()) && owner(e.Name(), stems) == stem(me) {
			out = append(out, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(out)
	return out
}

// VideoForSubtitle finds the video a sidecar subtitle belongs to.
func (l *Library) VideoForSubtitle(sub string) string {
	dir := filepath.Dir(sub)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	by := l.videosByStem(entries)
	stems := make([]string, 0, len(by))
	for st := range by {
		stems = append(stems, st)
	}
	if o := owner(filepath.Base(sub), stems); o != "" {
		return filepath.Join(dir, by[o])
	}
	return ""
}

// Signature changes whenever the video or any of its sidecar subtitles change.
func (l *Library) Signature(video string) string {
	st, err := os.Stat(video)
	if err != nil {
		return ""
	}
	h := sha1.New()
	fmt.Fprintf(h, "%d|%d", st.Size(), st.ModTime().UnixNano())
	for _, s := range l.Sidecars(video) {
		if ss, err := os.Stat(s); err == nil {
			fmt.Fprintf(h, "|%s|%d|%d", filepath.Base(s), ss.Size(), ss.ModTime().UnixNano())
		}
	}
	return hex.EncodeToString(h.Sum(nil))[:20]
}

// WalkVideos calls fn for every video under root (root may also be a single file).
func (l *Library) WalkVideos(root string, fn func(path string) error) error {
	st, err := os.Stat(root)
	if err != nil {
		return err
	}
	if !st.IsDir() {
		if l.IsVideo(root) {
			return fn(root)
		}
		return nil
	}
	return filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // unreadable dir: skip, keep going
		}
		if d.IsDir() {
			if p != root && (strings.HasPrefix(d.Name(), ".") || l.Ignored(d.Name())) {
				return filepath.SkipDir
			}
			return nil
		}
		if l.IsVideo(d.Name()) {
			return fn(p)
		}
		return nil
	})
}
