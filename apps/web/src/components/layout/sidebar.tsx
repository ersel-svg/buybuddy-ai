"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Video,
  Package,
  Database,
  GitCompare,
  Sparkles,
  Brain,
  Layers,
  Settings,
  ChevronLeft,
  ChevronRight,
  ImageIcon,
  Triangle,
  ScanLine,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// Navigation groups for different workflows
const navigationGroups = [
  {
    title: "Overview",
    items: [
      {
        name: "Dashboard",
        href: "/",
        icon: LayoutDashboard,
      },
    ],
  },
  {
    title: "Real-Synthetic Matching",
    items: [
      {
        name: "Videos",
        href: "/videos",
        icon: Video,
      },
      {
        name: "Products",
        href: "/products",
        icon: Package,
      },
      {
        name: "Scan Requests",
        href: "/scan-requests",
        icon: ScanLine,
      },
      {
        name: "Cutouts",
        href: "/cutouts",
        icon: ImageIcon,
      },
      {
        name: "Embeddings",
        href: "/embeddings",
        icon: Layers,
      },
      {
        name: "Matching",
        href: "/matching",
        icon: GitCompare,
      },
    ],
  },
  {
    title: "Model Training",
    items: [
      {
        name: "Datasets",
        href: "/datasets",
        icon: Database,
      },
      {
        name: "Triplets",
        href: "/triplets",
        icon: Triangle,
      },
      {
        name: "Training",
        href: "/training",
        icon: Brain,
      },
      {
        name: "Augmentation",
        href: "/augmentation",
        icon: Sparkles,
      },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  return (
    <TooltipProvider delayDuration={0}>
      <aside
        className={cn(
          "flex flex-col h-screen bg-card border-r transition-all duration-300",
          collapsed ? "w-16" : "w-64"
        )}
      >
        {/* Logo */}
        <div className="flex items-center h-16 px-4 border-b">
          <Link href="/" className="flex items-center">
            {collapsed ? (
              <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                <span className="text-primary-foreground font-bold text-sm">B</span>
              </div>
            ) : (
              <Image
                src="/logo.svg"
                alt="BuyBuddy"
                width={140}
                height={26}
                priority
              />
            )}
          </Link>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-2 space-y-4 overflow-y-auto">
          {navigationGroups.map((group) => (
            <div key={group.title}>
              {!collapsed && (
                <h3 className="px-3 mb-1 text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  {group.title}
                </h3>
              )}
              <div className="space-y-1">
                {group.items.map((item) => {
                  const isActive =
                    pathname === item.href ||
                    (item.href !== "/" && pathname.startsWith(item.href));

                  const NavLink = (
                    <Link
                      key={item.name}
                      href={item.href}
                      className={cn(
                        "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                        isActive
                          ? "bg-primary text-primary-foreground"
                          : "text-muted-foreground hover:text-foreground hover:bg-accent"
                      )}
                    >
                      <item.icon className="w-5 h-5 flex-shrink-0" />
                      {!collapsed && <span>{item.name}</span>}
                    </Link>
                  );

                  if (collapsed) {
                    return (
                      <Tooltip key={item.name}>
                        <TooltipTrigger asChild>{NavLink}</TooltipTrigger>
                        <TooltipContent side="right">{item.name}</TooltipContent>
                      </Tooltip>
                    );
                  }

                  return NavLink;
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* Footer */}
        <div className="p-2 border-t">
          {collapsed ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <Link
                  href="/settings"
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                    pathname === "/settings"
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground hover:bg-accent"
                  )}
                >
                  <Settings className="w-5 h-5" />
                </Link>
              </TooltipTrigger>
              <TooltipContent side="right">Settings</TooltipContent>
            </Tooltip>
          ) : (
            <Link
              href="/settings"
              className={cn(
                "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                pathname === "/settings"
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent"
              )}
            >
              <Settings className="w-5 h-5" />
              <span>Settings</span>
            </Link>
          )}

          {/* Collapse toggle */}
          <Button
            variant="ghost"
            size="sm"
            className="w-full mt-2 justify-center"
            onClick={() => setCollapsed(!collapsed)}
          >
            {collapsed ? (
              <ChevronRight className="w-4 h-4" />
            ) : (
              <>
                <ChevronLeft className="w-4 h-4 mr-2" />
                <span>Collapse</span>
              </>
            )}
          </Button>
        </div>
      </aside>
    </TooltipProvider>
  );
}
