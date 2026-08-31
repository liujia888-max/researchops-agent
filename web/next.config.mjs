/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  eslint: {
    // Linting is owned by the root repo (ruff for Python); skip ESLint so the
    // web build doesn't require an eslint toolchain this repo doesn't ship.
    ignoreDuringBuilds: true,
  },
};

export default nextConfig;
