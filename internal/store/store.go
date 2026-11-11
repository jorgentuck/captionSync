// Package store tracks every video seen in SQLite and doubles as the persistent job queue.
package store

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite" // pure-Go SQLite driver
)

const (
	Queued     = "queued"
	Processing = "processing"
	Fixed      = "fixed"     // timing corrected and written
	InSync     = "in_sync"   // checked, no change needed
	NeedsFix   = "needs_fix" // dry-run: would have been fixed
	Uncertain  = "uncertain" // could not determine timing confidently
	NoSubs     = "no_subs"   // no subtitles at all
	Embedded   = "embedded"  // only embedded subtitles, not checked (EMBEDDED_SUBS=report)
	Failed     = "failed"
	Missing    = "missing"
	New        = "new" // dequeued before processing
)

var terminal = map[string]bool{Fixed: true, InSync: true, NeedsFix: true, Uncertain: true, NoSubs: true, Embedded: true, Failed: true, Missing: true}

// AllStatuses is the display order used by the UI.
var AllStatuses = []string{Queued, Processing, Fixed, InSync, NeedsFix, Uncertain, Embedded, NoSubs, Failed, Missing, New}

func IsStatus(s string) bool {
	for _, x := range AllStatuses {
		if x == s {
			return true
		}
	}
	return false
}

const schema = `
CREATE TABLE IF NOT EXISTS files(
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'new',
    prev_status TEXT,
    signature TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    message TEXT NOT NULL DEFAULT '',
    offset_ms INTEGER,
    details TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    queued_at REAL, started_at REAL, finished_at REAL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_updated ON files(updated_at);
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    ts REAL NOT NULL,
    status TEXT NOT NULL,
    message TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_history_file ON history(file_id);
`

type File struct {
	ID         int64           `json:"id"`
	Path       string          `json:"path"`
	Status     string          `json:"status"`
	Signature  string          `json:"signature,omitempty"`
	Reason     string          `json:"reason"`
	Message    string          `json:"message"`
	OffsetMs   *int64          `json:"offset_ms"`
	Details    json.RawMessage `json:"details,omitempty"`
	Attempts   int             `json:"attempts"`
	QueuedAt   *float64        `json:"queued_at"`
	StartedAt  *float64        `json:"started_at"`
	FinishedAt *float64        `json:"finished_at"`
	UpdatedAt  float64         `json:"updated_at"`
}

type HistoryEntry struct {
	TS      float64 `json:"ts"`
	Status  string  `json:"status"`
	Message string  `json:"message"`
}

type Store struct {
	db   *sql.DB
	mu   sync.Mutex
	wake chan struct{}
}

func now() float64 { return float64(time.Now().UnixNano()) / 1e9 }

func Open(path string) (*Store, error) {
	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=journal_mode(WAL)&_pragma=busy_timeout(15000)&_pragma=synchronous(NORMAL)")
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(schema); err != nil {
		return nil, fmt.Errorf("init schema: %w", err)
	}
	return &Store{db: db, wake: make(chan struct{}, 1)}, nil
}

func (s *Store) Close() error { return s.db.Close() }

func (s *Store) notify() {
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

// Wake returns a channel that receives when new work is queued.
func (s *Store) Wake() <-chan struct{} { return s.wake }

func (s *Store) history(id int64, status, msg string) {
	_, _ = s.db.Exec(`INSERT INTO history(file_id, ts, status, message) VALUES(?,?,?,?)`, id, now(), status, msg)
}

// Enqueue queues a file. Result: "queued", "skipped" (handled before, unchanged) or "already".
func (s *Store) Enqueue(path, signature, reason string, force bool) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t := now()
	var id int64
	var status, sig string
	err := s.db.QueryRow(`SELECT id, status, signature FROM files WHERE path=?`, path).Scan(&id, &status, &sig)
	switch {
	case err == sql.ErrNoRows:
		res, err := s.db.Exec(`INSERT INTO files(path, status, reason, queued_at, created_at, updated_at) VALUES(?,?,?,?,?,?)`,
			path, Queued, reason, t, t, t)
		if err != nil {
			return "", err
		}
		id, _ = res.LastInsertId()
	case err != nil:
		return "", err
	default:
		if status == Queued || status == Processing {
			return "already", nil
		}
		if !force && terminal[status] && sig == signature {
			return "skipped", nil
		}
		if _, err := s.db.Exec(`UPDATE files SET status=?, prev_status=?, reason=?, queued_at=?, updated_at=? WHERE id=?`,
			Queued, status, reason, t, t, id); err != nil {
			return "", err
		}
	}
	s.history(id, Queued, reason)
	s.notify()
	return "queued", nil
}

// ClaimNext atomically takes the oldest queued file.
func (s *Store) ClaimNext() (*File, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var id int64
	var path, reason string
	err := s.db.QueryRow(`SELECT id, path, reason FROM files WHERE status=? ORDER BY queued_at, id LIMIT 1`, Queued).
		Scan(&id, &path, &reason)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	t := now()
	if _, err := s.db.Exec(`UPDATE files SET status=?, started_at=?, attempts=attempts+1, updated_at=? WHERE id=?`,
		Processing, t, t, id); err != nil {
		return nil, err
	}
	s.history(id, Processing, "")
	return &File{ID: id, Path: path, Reason: reason, Status: Processing}, nil
}

func (s *Store) Finish(id int64, status, message, signature string, offsetMs *int64, details any) error {
	var d any
	if details != nil {
		b, _ := json.Marshal(details)
		d = string(b)
	}
	t := now()
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.db.Exec(`UPDATE files SET status=?, message=?, signature=?, offset_ms=?, details=?, finished_at=?,
		updated_at=?, prev_status=NULL WHERE id=?`, status, message, signature, offsetMs, d, t, t, id)
	s.history(id, status, message)
	return err
}

// ResetProcessing re-queues anything interrupted by a restart.
func (s *Store) ResetProcessing() (int64, error) {
	res, err := s.db.Exec(`UPDATE files SET status=? WHERE status=?`, Queued, Processing)
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}

// ClearQueue drops everything waiting, restoring the status each file had before.
func (s *Store) ClearQueue() (int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	res, err := s.db.Exec(`UPDATE files SET status=COALESCE(prev_status, ?), prev_status=NULL WHERE status=?`, New, Queued)
	if err != nil {
		return 0, err
	}
	return res.RowsAffected()
}

func (s *Store) MarkMissing(path string) {
	var id int64
	var status, sig string
	if err := s.db.QueryRow(`SELECT id, status, signature FROM files WHERE path=?`, path).Scan(&id, &status, &sig); err == nil && status != Missing {
		_ = s.Finish(id, Missing, "File no longer exists", sig, nil, nil)
	}
}

const fileCols = `id, path, status, signature, reason, message, offset_ms, details, attempts, queued_at, started_at, finished_at, updated_at`

func scanFile(sc interface{ Scan(...any) error }) (*File, error) {
	var f File
	var off sql.NullInt64
	var det sql.NullString
	var q, st, fin sql.NullFloat64
	if err := sc.Scan(&f.ID, &f.Path, &f.Status, &f.Signature, &f.Reason, &f.Message, &off, &det, &f.Attempts,
		&q, &st, &fin, &f.UpdatedAt); err != nil {
		return nil, err
	}
	if off.Valid {
		f.OffsetMs = &off.Int64
	}
	if det.Valid && det.String != "" {
		f.Details = json.RawMessage(det.String)
	}
	opt := func(n sql.NullFloat64) *float64 {
		if n.Valid {
			return &n.Float64
		}
		return nil
	}
	f.QueuedAt, f.StartedAt, f.FinishedAt = opt(q), opt(st), opt(fin)
	return &f, nil
}

func (s *Store) Get(id int64) (*File, error) {
	f, err := scanFile(s.db.QueryRow(`SELECT `+fileCols+` FROM files WHERE id=?`, id))
	if err == sql.ErrNoRows {
		return nil, nil
	}
	return f, err
}

// StatusOf returns the recorded status of a path, or "" if never seen.
func (s *Store) StatusOf(path string) string {
	var st string
	_ = s.db.QueryRow(`SELECT status FROM files WHERE path=?`, path).Scan(&st)
	return st
}

func (s *Store) History(id int64, limit int) ([]HistoryEntry, error) {
	rows, err := s.db.Query(`SELECT ts, status, message FROM history WHERE file_id=? ORDER BY id DESC LIMIT ?`, id, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []HistoryEntry{}
	for rows.Next() {
		var h HistoryEntry
		if err := rows.Scan(&h.TS, &h.Status, &h.Message); err != nil {
			return nil, err
		}
		out = append(out, h)
	}
	return out, rows.Err()
}

func (s *Store) PathsWithStatus(statuses []string) ([]string, error) {
	if len(statuses) == 0 {
		return nil, nil
	}
	args := make([]any, len(statuses))
	for i, st := range statuses {
		args[i] = st
	}
	rows, err := s.db.Query(`SELECT path FROM files WHERE status IN (?`+strings.Repeat(",?", len(statuses)-1)+`) ORDER BY path`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			return nil, err
		}
		out = append(out, p)
	}
	return out, rows.Err()
}

func (s *Store) Stats() (map[string]int, error) {
	rows, err := s.db.Query(`SELECT status, COUNT(*) FROM files GROUP BY status`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := map[string]int{}
	for rows.Next() {
		var st string
		var n int
		if err := rows.Scan(&st, &n); err != nil {
			return nil, err
		}
		out[st] = n
	}
	return out, rows.Err()
}

type QueueItem struct {
	ID       int64   `json:"id"`
	Path     string  `json:"path"`
	Reason   string  `json:"reason"`
	QueuedAt float64 `json:"queued_at"`
}

func (s *Store) Queue(limit int) ([]QueueItem, error) {
	rows, err := s.db.Query(`SELECT id, path, reason, queued_at FROM files WHERE status=? ORDER BY queued_at, id LIMIT ?`, Queued, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []QueueItem{}
	for rows.Next() {
		var q QueueItem
		if err := rows.Scan(&q.ID, &q.Path, &q.Reason, &q.QueuedAt); err != nil {
			return nil, err
		}
		out = append(out, q)
	}
	return out, rows.Err()
}

func (s *Store) List(status, q string, limit, offset int) ([]*File, int, error) {
	var where []string
	var args []any
	if status != "" {
		where = append(where, "status=?")
		args = append(args, status)
	}
	if q != "" {
		where = append(where, "path LIKE ?")
		args = append(args, "%"+q+"%")
	}
	w := ""
	if len(where) > 0 {
		w = "WHERE " + strings.Join(where, " AND ")
	}
	var total int
	if err := s.db.QueryRow(`SELECT COUNT(*) FROM files `+w, args...).Scan(&total); err != nil {
		return nil, 0, err
	}
	rows, err := s.db.Query(`SELECT `+fileCols+` FROM files `+w+` ORDER BY updated_at DESC LIMIT ? OFFSET ?`,
		append(args, limit, offset)...)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	out := []*File{}
	for rows.Next() {
		f, err := scanFile(rows)
		if err != nil {
			return nil, 0, err
		}
		f.Details = nil // keep list responses small
		out = append(out, f)
	}
	return out, total, rows.Err()
}
