set -ex

npm install -g git-release-notes

# Get tags
CURRENT_TAG=$(git describe --tags --abbrev=0)
PRETTY_NAME="Release ${CURRENT_TAG#v}"
PREV_TAG=$(git describe --abbrev=0 --tags `git rev-list --tags --skip=1  --max-count=1`)

# Create release note
git-release-notes ${PREV_TAG}..${CURRENT_TAG} .github/release-template.ejs > ./CHANGELOG.tmp

# Create git release
gh release create $CURRENT_TAG -t "$PRETTY_NAME" -n "$(cat ./CHANGELOG.tmp)"
