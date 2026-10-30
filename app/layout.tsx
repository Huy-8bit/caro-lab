import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const title = "Caro Lab — Self-Learning Chess AI";
const description = "Train, watch, replay, and compare AlphaZero-inspired chess agents with real self-play data.";

export const metadata: Metadata = {
  title,
  description,
  icons: { icon: "/favicon.png", shortcut: "/favicon.png" },
  openGraph: { title, description, type: "website", images: [{ url: "/og.png", width: 2048, height: 1024, alt: "Caro Lab self-learning chess AI" }] },
  twitter: { card: "summary_large_image", title, description, images: ["/og.png"] },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
