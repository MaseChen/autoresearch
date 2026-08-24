import { createHash } from 'node:crypto'
import { readFileSync, writeFileSync } from 'node:fs'
import { spawnSync } from 'node:child_process'

const lock = readFileSync(new URL('../package-lock.json', import.meta.url))
const lockDigest = createHash('sha256').update(lock).digest('hex')
const npmCli = process.env.npm_execpath

if (!npmCli) {
  throw new Error('npm_execpath is required for a pinned SBOM generator')
}

const generated = spawnSync(
  process.execPath,
  [npmCli, 'sbom', '--package-lock-only', '--sbom-format', 'spdx'],
  { encoding: 'utf8', maxBuffer: 8 * 1024 * 1024 },
)

if (generated.status !== 0 || generated.stderr || !generated.stdout) {
  throw new Error(`npm sbom failed closed (status=${String(generated.status)})`)
}

const sbom = JSON.parse(generated.stdout)
sbom.documentNamespace = `https://github.com/masechen/autoresearch/sbom/console-v1/${lockDigest}`
sbom.creationInfo.created = '1970-01-01T00:00:00.000Z'
sbom.creationInfo.comment = `Deterministic package-lock SHA-256: ${lockDigest}`
writeFileSync(
  new URL('../sbom.spdx.json', import.meta.url),
  `${JSON.stringify(sbom, null, 2)}\n`,
  { encoding: 'utf8', mode: 0o644 },
)

console.log(`CONSOLE_SBOM=PASSED lock_sha256=${lockDigest}`)
