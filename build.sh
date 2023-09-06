commit_id="$(git rev-parse HEAD)"
registry="cs-ai.tencentcloudcr.com"
#registry="docker.io"
#repository="thexjx/traefik-with-plugins"
tag="${commit_id}"
platform="amd64" # arm64 or amd64

function build() {
    repository="triton/openplayground-${platform}"
    image_name="${registry}/${repository}:${tag}"
    echo image_name: "${image_name}"
    docker build --no-cache --platform linux/${platform} -t "${image_name}" -f Dockerfile .
    docker push "${image_name}"
}


build