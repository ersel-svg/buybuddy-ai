"use client";

import { useState, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { Eye, EyeOff, Loader2 } from "lucide-react";
import { motion, useMotionValue, useSpring } from "framer-motion";
import Image from "next/image";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { login } from "@/lib/auth";

// Generate particle positions once
const generateParticles = () =>
  Array.from({ length: 20 }, (_, i) => ({
    id: i,
    left: Math.random() * 100,
    top: Math.random() * 100,
    duration: 4 + Math.random() * 3,
    delay: Math.random() * 2,
  }));

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isMounted, setIsMounted] = useState(false);

  // Generate particles only once on client
  const particles = useMemo(() => generateParticles(), []);

  // Mouse tracking
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);
  const smoothMouseX = useSpring(mouseX, { damping: 30, stiffness: 100 });
  const smoothMouseY = useSpring(mouseY, { damping: 30, stiffness: 100 });

  useEffect(() => {
    setIsMounted(true);

    const handleMouseMove = (e: MouseEvent) => {
      mouseX.set(e.clientX);
      mouseY.set(e.clientY);
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, [mouseX, mouseY]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      await login({ username, password });
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden bg-black p-4">
      {/* Mouse Following Spotlight */}
      <motion.div
        className="pointer-events-none absolute w-[600px] h-[600px] rounded-full opacity-40 -translate-x-1/2 -translate-y-1/2"
        style={{
          background: "radial-gradient(circle, #335CFF 0%, transparent 60%)",
          filter: "blur(100px)",
          left: smoothMouseX,
          top: smoothMouseY,
        }}
      />

      {/* Animated Blue Orbs */}
      <motion.div
        className="absolute w-[500px] h-[500px] rounded-full opacity-25"
        style={{
          background: "radial-gradient(circle, #335CFF 0%, transparent 70%)",
          filter: "blur(80px)",
        }}
        animate={{
          x: ["-30%", "30%", "-30%"],
          y: ["-20%", "20%", "-20%"],
        }}
        transition={{
          duration: 25,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      <motion.div
        className="absolute w-[600px] h-[600px] rounded-full opacity-20"
        style={{
          background: "radial-gradient(circle, #335CFF 0%, transparent 70%)",
          filter: "blur(90px)",
        }}
        animate={{
          x: ["40%", "-20%", "40%"],
          y: ["30%", "-10%", "30%"],
        }}
        transition={{
          duration: 30,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      <motion.div
        className="absolute w-[450px] h-[450px] rounded-full opacity-30"
        style={{
          background: "radial-gradient(circle, #335CFF 0%, transparent 70%)",
          filter: "blur(70px)",
        }}
        animate={{
          x: ["-10%", "50%", "-10%"],
          y: ["50%", "-20%", "50%"],
        }}
        transition={{
          duration: 35,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      <motion.div
        className="absolute w-[550px] h-[550px] rounded-full opacity-15"
        style={{
          background: "radial-gradient(circle, #1e40af 0%, transparent 70%)",
          filter: "blur(85px)",
        }}
        animate={{
          x: ["60%", "-30%", "60%"],
          y: ["-30%", "40%", "-30%"],
        }}
        transition={{
          duration: 28,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Floating Particles */}
      {isMounted &&
        particles.map((particle) => (
          <motion.div
            key={particle.id}
            className="absolute w-1 h-1 bg-white/40 rounded-full"
            style={{
              left: `${particle.left}%`,
              top: `${particle.top}%`,
            }}
            animate={{
              y: [0, -40, 0],
              opacity: [0.2, 0.6, 0.2],
            }}
            transition={{
              duration: particle.duration,
              repeat: Infinity,
              delay: particle.delay,
              ease: "easeInOut",
            }}
          />
        ))}

      {/* Subtle Light Beams */}
      <motion.div
        className="absolute top-0 left-1/4 w-px h-full bg-gradient-to-b from-transparent via-blue-500/20 to-transparent"
        animate={{
          opacity: [0.1, 0.3, 0.1],
          scaleY: [0.8, 1, 0.8],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      <motion.div
        className="absolute top-0 right-1/3 w-px h-full bg-gradient-to-b from-transparent via-blue-500/20 to-transparent"
        animate={{
          opacity: [0.2, 0.4, 0.2],
          scaleY: [0.9, 1, 0.9],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 1,
        }}
      />

      {/* Login Card */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="relative z-10 w-full max-w-md"
      >
        <Card className="border-white/10 bg-black/40 backdrop-blur-xl shadow-2xl">
          <CardHeader className="text-center space-y-6 pb-6">
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.1 }}
              className="flex justify-center"
            >
              <Image
                src="/logo.svg"
                alt="BuyBuddy AI"
                width={180}
                height={34}
                priority
                className="h-8 w-auto"
              />
            </motion.div>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
              className="space-y-2"
            >
              <CardTitle className="text-xl font-semibold text-white">
                Welcome back
              </CardTitle>
              <CardDescription className="text-sm text-zinc-400">
                Sign in to your account
              </CardDescription>
            </motion.div>
          </CardHeader>
          <CardContent className="space-y-4">
            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  className="p-3 text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-md"
                >
                  {error}
                </motion.div>
              )}

              <div className="space-y-2">
                <Label htmlFor="username" className="text-sm text-zinc-300">
                  Username
                </Label>
                <Input
                  id="username"
                  type="text"
                  placeholder="Enter your username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  disabled={isLoading}
                  autoComplete="username"
                  className="bg-white/5 border-white/10 text-white placeholder:text-zinc-500 focus:border-[#335CFF]/50 focus:ring-[#335CFF]/20"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="password" className="text-sm text-zinc-300">
                  Password
                </Label>
                <div className="relative">
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    disabled={isLoading}
                    autoComplete="current-password"
                    className="bg-white/5 border-white/10 text-white placeholder:text-zinc-500 focus:border-[#335CFF]/50 focus:ring-[#335CFF]/20 pr-10"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500 hover:text-zinc-300 transition-colors"
                    tabIndex={-1}
                  >
                    {showPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>

              <Button
                type="submit"
                className="w-full bg-[#335CFF] hover:bg-[#335CFF]/90 text-white"
                disabled={isLoading}
              >
                {isLoading ? (
                  <span className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Signing in...
                  </span>
                ) : (
                  "Sign in"
                )}
              </Button>
            </form>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
