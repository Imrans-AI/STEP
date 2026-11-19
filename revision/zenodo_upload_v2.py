"""
Upload the STEP revision deposit to an existing Zenodo draft.

    ZENODO_TOKEN=... python3 -I revision/zenodo_upload_v2.py
    python3 -I revision/zenodo_upload_v2.py --dry-run     # no auth, local checks only

Zenodo's bucket API rejects a '/' in an object key, so the staged directory tree
cannot go up file-by-file. The tree travels as one archive instead -
STEP_revision_v2.zip, internal structure preserved - with README.md and
MANIFEST.tsv also at the root of the deposit so both are readable on the record
page without downloading 825 MB.

The token is read from the environment only. It is never printed, never written
to a file, never placed in a URL query string: it travels in an Authorization
header, and every string printed passes through scrub() first, so a token echoed
back inside an API error body cannot reach the log.

This script does NOT publish. It leaves a draft and prints its URL.
Re-running is safe: a key already in the bucket with a matching MD5 is skipped,
so an interrupted 825 MB transfer restarts without re-sending what landed.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

import requests

BASE = 'https://zenodo.org'
REPO = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
STAGING = Path('/home/muhammad-imran/Downloads/STEP_zenodo_v2_staging')
ARCHIVE = Path('/home/muhammad-imran/Downloads/STEP_revision_v2.zip')
DRAFT = 24359126        # the draft already cut from this lineage; reused, not replaced
RECORD = 21473727       # only consulted by --new-version
VERSION = '2.0.0'
NOTE_MARK = 'Revision deposit (v2.0.0)'
NOTES = ('Revision deposit for Protein Science manuscript 1062536: corrected '
         'contact prior, whitespace-normalised splits, similarity-disjoint '
         'splits, eighteen retrained checkpoints (fifteen for the '
         'prior-control matrix, three for the similarity-disjoint split) and '
         'corrected baseline predictions. See README.md.')

# Files that go up as their own root-level objects, in upload order.
ROOT_FILES = [
    (ARCHIVE, 'STEP_revision_v2.zip'),
    (STAGING / 'README.md', 'README.md'),
    (STAGING / 'MANIFEST.tsv', 'MANIFEST.tsv'),
]

_TOKEN = os.environ.get('ZENODO_TOKEN', '')


def scrub(s):
    """Never let the token reach stdout, even inside an echoed error body."""
    s = str(s)
    if _TOKEN and len(_TOKEN) > 3:
        s = s.replace(_TOKEN, '<ZENODO_TOKEN redacted>')
    return s


def say(*a):
    print(scrub(' '.join(str(x) for x in a)), flush=True)


def hdr():
    return {'Authorization': 'Bearer ' + _TOKEN}


def api(method, url, *, ok=(200, 201, 202, 204), **kw):
    kw.setdefault('timeout', 300)
    h = kw.pop('headers', {})
    h.update(hdr())
    r = requests.request(method, url, headers=h, **kw)
    if r.status_code not in ok:
        detail = r.text[:2000]
        try:                                  # Zenodo returns a list of errors
            body = r.json()
            lines = ['%s: %s' % (e.get('field', '?'), e.get('message', e))
                     for e in body.get('errors', [])]
            if lines:
                detail = '%s\n  %s' % (body.get('message', ''), '\n  '.join(lines))
        except Exception:
            pass
        raise SystemExit(scrub(
            'HTTP %s on %s %s\n%s' % (r.status_code, method, url, detail)))
    return r


def digest(path, algo, chunk=1 << 22):
    h = hashlib.new(algo)
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()


def human(n):
    n = float(n)
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return '%.2f %s' % (n, u)
        n /= 1024.0


def git_head():
    try:
        return subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                                       text=True).strip()
    except Exception as e:
        raise SystemExit('could not resolve HEAD: %s' % e)


# --------------------------------------------------------------- the manifest
def read_manifest():
    """path -> (bytes, sha256), plus the commit the manifest was written at."""
    rows, commit = {}, None
    for line in (STAGING / 'MANIFEST.tsv').read_text().splitlines():
        if line.startswith('#'):
            if 'repository commit' in line:
                commit = line.split()[-1]
            continue
        if line.startswith('path\t'):
            continue
        p, b, s = line.split('\t')
        rows[p] = (int(b), s)
    return rows, commit


def verify_archive(manifest, deep):
    """The archive must hold exactly the staged tree, and match MANIFEST.tsv."""
    if not ARCHIVE.exists():
        raise SystemExit('archive missing: %s' % ARCHIVE)
    staged = sorted(p.relative_to(STAGING).as_posix()
                    for p in STAGING.rglob('*') if p.is_file())
    with zipfile.ZipFile(ARCHIVE) as z:
        names = sorted(i.filename for i in z.infolist() if not i.is_dir())
        raw = sum(i.file_size for i in z.infolist() if not i.is_dir())
        say('Archive :', ARCHIVE.name)
        say('  entries            : %d' % len(names))
        say('  uncompressed       : %s' % human(raw))
        say('  archive on disk    : %s  (%.1f%% of the tree)'
            % (human(ARCHIVE.stat().st_size),
               100.0 * ARCHIVE.stat().st_size / raw))
        if names != staged:
            only_z = set(names) - set(staged)
            only_s = set(staged) - set(names)
            raise SystemExit('archive does not match the staging tree; '
                             'in archive only: %s; on disk only: %s'
                             % (sorted(only_z)[:5], sorted(only_s)[:5]))
        say('  entry list         : identical to the staging tree')
        if 'MANIFEST.tsv' not in names or 'README.md' not in names:
            raise SystemExit('archive is missing README.md or MANIFEST.tsv')

        if deep:
            say('  reading every entry back and checking SHA-256 against '
                'MANIFEST.tsv ...')
            bad = []
            for key, (nbytes, sha) in manifest.items():
                h = hashlib.sha256()
                with z.open(key) as fh:
                    got = 0
                    for blk in iter(lambda: fh.read(1 << 22), b''):
                        h.update(blk)
                        got += len(blk)
                if got != nbytes or h.hexdigest() != sha:
                    bad.append(key)
            if bad:
                for k in bad[:10]:
                    say('  MISMATCH in archive:', k)
                raise SystemExit('%d archive entries disagree with MANIFEST.tsv'
                                 % len(bad))
            say('  all %d manifest entries read back correct from the archive'
                % len(manifest))
    return len(names), raw


# ------------------------------------------------------------------- transfer
class Progress:
    """Streams a file to requests while printing throughput."""

    def __init__(self, path, size, label):
        self.fh = open(path, 'rb')
        self.size = size
        self.label = label
        self.done = 0
        self.t0 = time.time()
        self.last = 0.0

    def __len__(self):
        return self.size

    def read(self, n=-1):
        blk = self.fh.read(n)
        self.done += len(blk)
        now = time.time()
        if now - self.last > 3.0 or self.done >= self.size:
            el = max(now - self.t0, 1e-6)
            pct = 100.0 * self.done / self.size if self.size else 100.0
            say('      %s  %5.1f%%  %s / %s  %.1f MB/s'
                % (self.label, pct, human(self.done), human(self.size),
                   self.done / el / 1e6))
            self.last = now
        return blk

    def close(self):
        self.fh.close()


def bucket_map(bucket):
    """key -> (md5 hex, size)."""
    out = {}
    for c in api('GET', bucket).json().get('contents', []):
        ck = c.get('checksum', '')
        out[c['key']] = (ck.split(':', 1)[1] if ':' in ck else ck,
                         int(c.get('size') or 0))
    return out


def bucket_contents(bucket):
    """key -> md5 hex."""
    return {k: v[0] for k, v in bucket_map(bucket).items()}


def put_file(bucket, path, key, size, tries=3):
    for attempt in range(1, tries + 1):
        pf = Progress(path, size, key)
        try:
            api('PUT', '%s/%s' % (bucket, key), data=pf, timeout=(30, 7200),
                headers={'Content-Length': str(size)})
            return time.time() - pf.t0
        except (requests.RequestException, SystemExit) as e:
            if attempt == tries:
                raise
            say('    attempt %d failed (%s); retrying in %ds'
                % (attempt, scrub(e)[:160], 10 * attempt))
            time.sleep(10 * attempt)
        finally:
            pf.close()


# ------------------------------------------------------------------- metadata
def dump_payload(meta):
    """Print the exact JSON that will be sent, in full."""
    say('')
    say('-' * 74)
    say('METADATA PAYLOAD  (sent as {"metadata": {...}})')
    say('-' * 74)
    for line in json.dumps(meta, indent=2, ensure_ascii=False).splitlines():
        say(line)
    say('-' * 74)
    say('field shapes:')
    for k in ('resource_type', 'upload_type', 'license', 'access_right',
              'version', 'publication_date', 'notes'):
        if k in meta:
            say('  %-18s %-7s %s' % (k, type(meta[k]).__name__,
                                     json.dumps(meta[k])[:90]))
    say('  %-18s %-7s %d entries' % ('creators', 'list', len(meta.get('creators', []))))
    say('  %-18s %-7s %d chars' % ('description', 'str', len(meta.get('description', ''))))
    say('-' * 74)


RESOURCE_TYPE = 'dataset'


def _flatten(v, keys):
    """A records-API object such as {"type": "dataset"} -> "dataset"."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in keys:
            if isinstance(v.get(k), str):
                return v[k]
    return None


def to_legacy(m, rt_as):
    """Coerce records-API metadata into the shape the deposit API validates.

    The records API serialises resource_type as {"title": ..., "type": ...} and
    license as {"id": ...}. The legacy deposit schema validates both as plain
    strings - it answered "metadata.resource_type: Not a valid string" to the
    object form, which is marshmallow's String-field error for a dict. So both
    are flattened here. rt_as leaves the choice reversible in one flag.
    """
    m = json.loads(json.dumps(m))
    for k in ('doi', 'prereserve_doi', 'relations', 'conceptdoi', 'conceptrecid'):
        m.pop(k, None)

    lic = _flatten(m.get('license'), ('id', 'identifier'))
    if lic:
        m['license'] = lic

    rt = (_flatten(m.get('resource_type'), ('type', 'id'))
          or _flatten(m.get('upload_type'), ('type', 'id'))
          or RESOURCE_TYPE)
    m.pop('resource_type', None)
    m.pop('upload_type', None)
    if rt_as == 'string':
        m['resource_type'] = rt
    elif rt_as == 'object':
        m['resource_type'] = {'type': rt}
    else:                                       # 'upload_type'
        m['upload_type'] = rt
    return m


def build_metadata(base_meta, commit, rt_as, pubdate):
    """v1.2.0 title and creators kept verbatim; version bumped; note appended."""
    m = to_legacy(base_meta, rt_as)
    m['version'] = VERSION
    m['publication_date'] = pubdate
    m['notes'] = NOTES
    desc = m.get('description', '')
    tag = '<p><strong>%s' % NOTE_MARK
    if tag in desc:                      # replace any note left by an earlier run
        desc = desc[:desc.index(tag)]
    m['description'] = desc + (
        '<p><strong>%s.</strong> This version replaces the released files with '
        'the deposit assembled during the Protein Science revision (manuscript '
        '1062536): corrected structure priors and physicochemical maps, '
        'normalised and similarity-disjoint splits, the retrained checkpoints '
        'for every prior configuration and for the similarity-disjoint arm, the '
        'corrected baseline predictions, and every script written during the '
        'revision. <em>What changed from the released version, and why, is set '
        'out in README.md</em>, which is also at the root of this record so it '
        'can be read without downloading the archive. The files themselves are '
        'in <code>STEP_revision_v2.zip</code> with their directory structure '
        'preserved; per-file paths, sizes and SHA-256 checksums are in '
        'MANIFEST.tsv, likewise at the root. Code state: commit '
        '<code>%s</code>.</p>' % (NOTE_MARK, commit))
    return m


def latest_of(record_id):
    r = requests.get('%s/api/records/%d' % (BASE, record_id), timeout=60)
    r.raise_for_status()
    d = r.json()
    concept = d.get('conceptrecid') or d.get('conceptdoi', '').split('.')[-1]
    r = requests.get('%s/api/records/%s/versions/latest' % (BASE, concept),
                     timeout=60)
    r.raise_for_status()
    return d, r.json()


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='local checks and metadata preview only; no auth, no writes')
    ap.add_argument('--draft', type=int, default=DRAFT,
                    help='existing draft deposition to reuse (default %d)' % DRAFT)
    ap.add_argument('--new-version', action='store_true',
                    help='cut a fresh new-version draft instead of reusing --draft')
    ap.add_argument('--commit', default=None,
                    help="code state to name in the description; a hash, or "
                         "'HEAD'. Default: the commit MANIFEST.tsv records, "
                         "which is what built the staged content.")
    ap.add_argument('--publication-date', default=None,
                    help='YYYY-MM-DD for the deposit (default: today)')
    ap.add_argument('--shallow', action='store_true',
                    help='skip reading every archive entry back against MANIFEST.tsv')
    ap.add_argument('--metadata-only', action='store_true',
                    help='set metadata on the draft and upload nothing')
    ap.add_argument('--resource-type-as', default='string',
                    choices=('string', 'object', 'upload_type'),
                    help='how to encode the resource type; the legacy deposit '
                         'schema validates it as a string (default)')
    a = ap.parse_args()

    manifest, mcommit = read_manifest()
    if a.commit is None:
        commit = mcommit                       # what built the staged content
    elif a.commit.upper() == 'HEAD':
        commit = git_head()
    else:
        commit = a.commit
    pubdate = a.publication_date or date.today().isoformat()
    if a.metadata_only:
        say('Metadata-only run: the archive is not read, hashed or sent.')
        n_entries, raw = 0, 0
    else:
        n_entries, raw = verify_archive(manifest, deep=not a.shallow)

    say('')
    say('Commit for the description : %s%s'
        % (commit, '  (= MANIFEST.tsv)' if commit == mcommit else ''))
    say('Commit in MANIFEST.tsv     : %s' % mcommit)
    say('Repository HEAD            : %s' % git_head())
    say('Publication date           : %s' % pubdate)

    to_send = []
    for path, key in ([] if a.metadata_only else ROOT_FILES):
        if not path.exists():
            raise SystemExit('missing: %s' % path)
        if '/' in key:
            raise SystemExit('bucket keys must not contain "/": %s' % key)
        to_send.append((path, key, path.stat().st_size))
    if to_send:
        say('')
        say('Deposit layout (%d root-level objects, no "/" in any key):'
            % len(to_send))
        for _, key, size in to_send:
            say('  %-24s %s' % (key, human(size)))

    if a.dry_run:
        say('')
        say('No token used. Previewing the metadata payload from the published '
            'v1.2.0 record; a live run reads the draft\'s own metadata instead.')
        _, newest = latest_of(RECORD)
        dump_payload(build_metadata(newest['metadata'], commit, a.resource_type_as,
                                    pubdate))
        say('')
        say('DRY RUN: nothing was sent.')
        return

    if not _TOKEN:
        raise SystemExit(
            'ZENODO_TOKEN is not set in the environment.\n'
            'Run:  ZENODO_TOKEN=<token> python3 -I revision/zenodo_upload_v2.py')

    say('')
    if a.new_version:
        say('Cutting a new version from %d ...' % RECORD)
        _, newest = latest_of(RECORD)
        r = api('POST', '%s/api/deposit/depositions/%d/actions/newversion'
                % (BASE, newest['id']))
        draft = api('GET', r.json()['links']['latest_draft']).json()
    else:
        say('Reusing draft deposition %d ...' % a.draft)
        draft = api('GET', '%s/api/deposit/depositions/%d' % (BASE, a.draft)).json()

    dep_id = draft['id']
    state = draft.get('state')
    say('  deposition %s, state=%s, submitted=%s'
        % (dep_id, state, draft.get('submitted')))
    if state == 'done':
        raise SystemExit('deposition %s is published, not a draft; refusing to touch it'
                         % dep_id)

    bucket = draft['links']['bucket']
    present = bucket_contents(bucket)
    say('  draft currently holds %d file(s): %s'
        % (len(present), ', '.join(sorted(present)) or '(none)'))

    say('')
    say('Uploading ...' if to_send else 'Uploading: nothing to send.')
    sent_bytes = skipped_bytes = 0
    sent = skipped = 0
    for path, key, size in to_send:
        local_md5 = digest(path, 'md5')
        if present.get(key) == local_md5:
            skipped += 1
            skipped_bytes += size
            say('  skip  %-24s %s  (already present, MD5 matches)'
                % (key, human(size)))
            continue
        say('  send  %-24s %s' % (key, human(size)))
        dt = put_file(bucket, path, key, size)
        sent += 1
        sent_bytes += size
        say('        done in %.1f s (%.1f MB/s)' % (dt, size / max(dt, 1e-6) / 1e6))

    say('')
    say('Verifying the draft ...' if to_send else 'Listing the draft ...')
    final_map = bucket_map(bucket)
    final = {k: v[0] for k, v in final_map.items()}
    final_sizes = sorted(((k, v[1]) for k, v in final_map.items()))
    bad = []
    for path, key, size in to_send:
        want = digest(path, 'md5')
        got = final.get(key)
        if got is None:
            bad.append((key, 'absent from the draft'))
        elif got != want:
            bad.append((key, 'server MD5 %s != local %s' % (got, want)))
        else:
            say('  %-24s MD5 %s  OK' % (key, got))
    if to_send:
        say('  archive SHA-256          %s' % digest(ARCHIVE, 'sha256'))
    for key, size in final_sizes:
        if key not in {k for _, k, _ in to_send}:
            say('  %-24s %s  already in the draft, not sent by this run'
                % (key, human(size)))

    say('')
    say('Building metadata ...')
    # The draft's own metadata is already in the shape the deposit API
    # validates, so it is the base; the published record is only a fallback.
    base = draft.get('metadata') or {}
    if base.get('title') and base.get('creators'):
        say('  base: the draft\'s own metadata (deposit API)')
        was = base.get('version')
    else:
        say('  base: the published v1.2.0 record (the draft carries no metadata)')
        _, newest = latest_of(RECORD)
        base = newest['metadata']
        was = base.get('version')
    meta = build_metadata(base, commit, a.resource_type_as, pubdate)

    dump_payload(meta)

    say('')
    say('Sending metadata ...')
    api('PUT', '%s/api/deposit/depositions/%s' % (BASE, dep_id),
        json={'metadata': meta}, headers={'Content-Type': 'application/json'})
    say('  accepted')
    say('  title    : %s' % meta['title'])
    say('  creators : %d' % len(meta['creators']))
    say('  version  : %s  (was %s)' % (meta['version'], was))
    say('  note     : names README.md, STEP_revision_v2.zip, MANIFEST.tsv, %s'
        % commit)

    say('')
    say('=' * 74)
    if to_send:
        say('archive       : %d entries, %s uncompressed, %s on disk'
            % (n_entries, human(raw), human(ARCHIVE.stat().st_size)))
    else:
        say('archive       : not touched (metadata-only run)')
    say('uploaded      : %d object(s), %s' % (sent, human(sent_bytes)))
    say('skipped       : %d object(s), %s (already present, MD5 matched)'
        % (skipped, human(skipped_bytes)))
    say('draft holds   : %s across %d object(s)'
        % (human(sum(c[1] for c in final_sizes)), len(final_sizes)))
    say('checksums     : %s' % ('ALL MATCH' if not bad else '%d MISMATCH' % len(bad)))
    for k, why in bad:
        say('  MISMATCH', k, why)
    say('state         : DRAFT, not published')
    say('draft URL     : %s/uploads/%s' % (BASE, dep_id))
    say('              : %s/deposit/%s   (legacy editor)' % (BASE, dep_id))
    say('=' * 74)
    if bad:
        sys.exit(1)


if __name__ == '__main__':
    main()
