const [major, minor] = process.versions.node.split('.').map(Number)

if (major !== 24 || minor < 11) {
  console.error(`Console release requires Node 24.11+, observed ${process.versions.node}`)
  process.exit(1)
}

console.log(`NODE_RELEASE_PRECHECK=PASSED node=${process.versions.node}`)
