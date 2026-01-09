import React from "react";

export default function Header() {
  return (
    <header style={styles.header}>
      <h1 style={styles.title}>iSAM</h1>
      <p style={styles.subtitle}>
        interactive Spike Activity Mapper
      </p>
    </header>
  );
}

const styles = {
  header: {
    textAlign: "center",
    marginBottom: "2.5rem",
  },
  title: {
    fontSize: "3.2rem",
    fontWeight: 700,
    margin: 0,
    color: "#1f2937",
    letterSpacing: "-0.02em",
  },
  subtitle: {
    marginTop: "0.35rem",
    fontSize: "1.05rem",
    fontWeight: 400,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
    color: "#6b7280",
  },
};
