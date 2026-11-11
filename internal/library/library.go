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

// MapPath translates a path as another container (e.g. Tdarr) sees it into ours.
func (l *Library) MapPath(p string) string {
	p = strings.TrimSpace(p)
	for _, m := range l.cfg.PathMappings {
		if p == m[0] || strings.HasPrefix(p, m[0]+"/") {
			return m[1] + p[len(m[0]):]
		}
	}
	return p
}

// Allowed reports whether p lies inside one of the configured media roots.
func (l *Library) Allowed(p string) bool {
	p = filepath.Clean(p)
	for _, r := range l.cfg.MediaRoots {
		if p == r || strings.HasPrefix(p, strings.TrimRight(r, "/")+"/") {
			return true
		}
	}
	return false
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

func (l *Library) Sidecars(video string) []string {
	dir := filepath.Dir(video)
	me := stem(filepath.Base(video))
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	stems := []string{me}
	for _, e := range entries {
		if !e.IsDir() && l.IsVideo(e.Name()) {
			stems = append(stems, stem(e.Name()))
		}
	}
	var out []string
	for _, e := range entries {
		if !e.IsDir() && l.IsSubtitle(e.Name()) && owner(e.Name(), stems) == me {
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
	byStem := map[string]string{}
	var stems []string
	for _, e := range entries {
		if !e.IsDir() && l.IsVideo(e.Name()) {
			byStem[stem(e.Name())] = e.Name()
			stems = append(stems, stem(e.Name()))
		}
	}
	if o := owner(filepath.Base(sub), stems); o != "" {
		return filepath.Join(dir, byStem[o])
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
