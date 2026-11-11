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

// Queue priorities: higher runs first, then oldest first.
const (
	PriorityLater = -1 // thorough re-checks of many files: after the bulk work (they're GPU-heavy)
	PriorityBulk  = 0  // "process all", scheduled / startup scans
	PriorityUser  = 1  // a folder or status re-check you started
	PriorityNow   = 2  // watcher, Bazarr/Tdarr hooks, a single file you picked
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
	if err := migrate(db); err != nil {
		return nil, fmt.Errorf("migrate: %w", err)
	}
	return &Store{db: db, wake: make(chan struct{}, 1)}, nil
}

// migrate upgrades databases created by older versions in place.
func migrate(db *sql.DB) error {
	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM pragma_table_info('files') WHERE name='priority'`).Scan(&n); err != nil {
		return err
	}
	if n == 0 {
		if _, err := db.Exec(`ALTER TABLE files ADD COLUMN priority INTEGER NOT NULL DEFAULT 0`); err != nil {
			return err
		}
	}
	if _, err := db.Exec(`CREATE INDEX IF NOT EXISTS idx_files_queue ON files(status, priority DESC, queued_at, id)`); err != nil {
		return err
	}
	var ver int
	if err := db.QueryRow(`PRAGMA user_version`).Scan(&ver); err != nil {
		return err
	}
	// Thorough re-checks queued for many files at once run after the bulk work (an older version
	// queued them ahead of it, where they hogged the GPU).
	if _, err := db.Exec(`UPDATE files SET priority=? WHERE status=? AND priority=1 AND reason LIKE 'thorough%'`,
		PriorityLater, Queued); err != nil {
		return err
	}
	if ver < 2 {
		// v2: 'attempts' now counts consecutive unfinished claims; older versions counted every run
		if _, err := db.Exec(`UPDATE files SET attempts=0`); err != nil {
			return err
		}
		if _, err := db.Exec(`PRAGMA user_version = 2`); err != nil {
			return err
		}
	}
	return nil
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

// Enqueue queues a file with a priority (PriorityBulk/User/Now).
// Result: "queued", "skipped" (handled before, unchanged) or "already" (waiting or running;
// a waiting file is bumped up if the new request has a higher priority).
func (s *Store) Enqueue(path, signature, reason string, force bool, priority int) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t := now()
	var id int64
	var status, sig string
	var prio int
	err := s.db.QueryRow(`SELECT id, status, signature, priority FROM files WHERE path=?`, path).Scan(&id, &status, &sig, &prio)
	switch {
	case err == sql.ErrNoRows:
		res, err := s.db.Exec(`INSERT INTO files(path, status, reason, priority, queued_at, created_at, updated_at) VALUES(?,?,?,?,?,?,?)`,
			path, Queued, reason, priority, t, t, t)
		if err != nil {
			return "", err
		}
		id, _ = res.LastInsertId()
	case err != nil:
		return "", err
	default:
		if status == Queued {
			if priority > prio {
				if _, err := s.db.Exec(`UPDATE files SET priority=?, reason=?, updated_at=? WHERE id=?`, priority, reason, t, id); err != nil {
					return "", err
				}
				s.history(id, Queued, reason+" (moved up the queue)")
				s.notify()
			}
			return "already", nil
		}
		if status == Processing {
			return "already", nil
		}
		if !force && terminal[status] && sig == signature {
			return "skipped", nil
		}
		if _, err := s.db.Exec(`UPDATE files SET status=?, prev_status=?, reason=?, priority=?, queued_at=?, updated_at=? WHERE id=?`,
			Queued, status, reason, priority, t, t, id); err != nil {
			return "", err
		}
	}
	s.history(id, Queued, reason)
	s.notify()
	return "queued", nil
}

// MaxAttempts: a file claimed this many times without ever finishing (crashes, restarts) is given up.
const MaxAttempts = 5

// ClaimNext atomically takes the next file: highest priority first, then oldest. With
// skipThorough, thorough re-checks are passed over (one is already running).
func (s *Store) ClaimNext(skipThorough ...bool) (*File, error) {
	skip := 0
	if len(skipThorough) > 0 && skipThorough[0] {
		skip = 1
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for {
		var id int64
		var path, reason string
		var attempts int
		err := s.db.QueryRow(`SELECT id, path, reason, attempts FROM files WHERE status=? AND queued_at <= ?
			AND (? = 0 OR reason NOT LIKE 'thorough%')
			ORDER BY priority DESC, queued_at, id LIMIT 1`, Queued, now(), skip).
			Scan(&id, &path, &reason, &attempts)
		if err == sql.ErrNoRows {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
		t := now()
		if attempts >= MaxAttempts {
			msg := fmt.Sprintf("gave up after %d attempts that never finished (crash, restart or hang)", attempts)
			if _, err := s.db.Exec(`UPDATE files SET status=?, message=?, attempts=0, finished_at=?, updated_at=?, prev_status=NULL WHERE id=?`,
				Failed, msg, t, t, id); err != nil {
				return nil, err
			}
			s.history(id, Failed, msg)
			continue
		}
		if _, err := s.db.Exec(`UPDATE files SET status=?, started_at=?, attempts=attempts+1, updated_at=? WHERE id=?`,
			Processing, t, t, id); err != nil {
			return nil, err
		}
		s.history(id, Processing, "")
		return &File{ID: id, Path: path, Reason: reason, Status: Processing}, nil
	}
}

// Finish records the result of a job. It only applies to a file that is still 'processing' (so a
// late result can never overwrite a newer state), and resets the unfinished-attempts counter.
func (s *Store) Finish(id int64, status, message, signature string, offsetMs *int64, details any) error {
	var d any
	if details != nil {
		b, _ := json.Marshal(details)
		d = string(b)
	}
	t := now()
	s.mu.Lock()
	defer s.mu.Unlock()
	res, err := s.db.Exec(`UPDATE files SET status=?, message=?, signature=?, offset_ms=?, details=?, finished_at=?,
		updated_at=?, prev_status=NULL, attempts=0 WHERE id=? AND status=?`, status, message, signature, offsetMs, d, t, t, id, Processing)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 1 {
		s.history(id, status, message)
	}
	return nil
}

// Requeue puts a file that is being processed back in the queue, behind the other files of its
// priority, not to be picked up for `delay`. countAttempt=false when it was not the file's fault.
func (s *Store) Requeue(id int64, message string, countAttempt bool, delay time.Duration) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	dec := 1
	if countAttempt {
		dec = 0
	}
	res, err := s.db.Exec(`UPDATE files SET status=?, message=?, attempts=MAX(0, attempts-?), queued_at=?, updated_at=? WHERE id=? AND status=?`,
		Queued, message, dec, now()+delay.Seconds(), now(), id, Processing)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 1 {
		s.history(id, Queued, message)
		s.notify()
	}
	return nil
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

// MarkMissing records that a file is gone - unless it is queued or being processed (a worker
// will find out itself; marking it here could let a second worker pick it up).
func (s *Store) MarkMissing(path string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var id int64
	var status string
	if err := s.db.QueryRow(`SELECT id, status FROM files WHERE path=?`, path).Scan(&id, &status); err != nil ||
		status == Missing || status == Queued || status == Processing {
		return
	}
	t := now()
	if _, err := s.db.Exec(`UPDATE files SET status=?, message=?, finished_at=?, updated_at=?, prev_status=NULL WHERE id=?`,
		Missing, "File no longer exists", t, t, id); err == nil {
		s.history(id, Missing, "File no longer exists")
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
	Priority int     `json:"priority"`
	QueuedAt float64 `json:"queued_at"`
}

func (s *Store) Queue(limit int) ([]QueueItem, error) {
	rows, err := s.db.Query(`SELECT id, path, reason, priority, queued_at FROM files WHERE status=? ORDER BY priority DESC, queued_at, id LIMIT ?`, Queued, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []QueueItem{}
	for rows.Next() {
		var q QueueItem
		if err := rows.Scan(&q.ID, &q.Path, &q.Reason, &q.Priority, &q.QueuedAt); err != nil {
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

// UncertainReasons counts uncertain files by why they are uncertain.
func (s *Store) UncertainReasons() map[string]int {
	out := map[string]int{}
	rows, err := s.db.Query(`SELECT COALESCE(json_extract(details, '$.reason'), 'other'), COUNT(*) FROM files
		WHERE status=? GROUP BY 1`, Uncertain)
	if err != nil {
		return out
	}
	defer rows.Close()
	for rows.Next() {
		var r string
		var n int
		if rows.Scan(&r, &n) == nil {
			out[r] = n
		}
	}
	return out
}

// ProcessingIDs lists files currently marked as being processed.
func (s *Store) ProcessingIDs() ([]int64, error) {
	rows, err := s.db.Query(`SELECT id FROM files WHERE status=?`, Processing)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		out = append(out, id)
	}
	return out, rows.Err()
}
