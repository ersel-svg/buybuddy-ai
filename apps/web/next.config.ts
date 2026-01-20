import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow external images
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "*.supabase.co",
      },
      {
        protocol: "https",
        hostname: "*.buybuddy.co",
      },
      {
        protocol: "https",
        hostname: "buybuddy.co",
      },
      {
        protocol: "https",
        hostname: "*.amazonaws.com",
      },
      {
        protocol: "http",
        hostname: "localhost",
      },
    ],
  },
  // Output standalone for production
  output: "standalone",
};

export default nextConfig;
