/** @type {import('next').NextConfig} */
const nextConfig = {
    pageExtensions: ['ts', 'tsx'],
    experimental: {
        // No ISR for local dev — always fresh reads from disk
        staleTimes: { dynamic: 0, static: 0 },
    },
}

export default nextConfig
