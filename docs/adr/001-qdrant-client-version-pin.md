# ADR-001: Qdrant Client Version Pin

**Status**: Active
**Date**: 2026-06-10

## Context
qdrant-client is pinned to `>=1.9.0,<1.10.0` in `requirements.txt` because the Qdrant server running on the homelab is on a version compatible with the 1.9.x client API. Upgrading the client without upgrading the server risks API incompatibilities.

## Decision
Keep the pin until the Qdrant server is upgraded. Client and server upgrades must be performed together in a single coordinated change.

## Upgrade path
Steps to perform the upgrade when ready:
1. Check the Qdrant release notes for breaking API changes between 1.9.x and the target version.
2. Upgrade the Qdrant server container/binary.
3. Update the pin in `requirements.txt`.
4. Run the full test suite.
5. Verify live search still works.

## Consequences
Occasional security patches or bug fixes in qdrant-client >=1.10 are not available until the coordinated upgrade is done.
