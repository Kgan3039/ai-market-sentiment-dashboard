import React from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import App from "./App";

const dataAsOf = "2026-07-15T15:30:00Z";

const tickers = [
  { ticker: "TSLA", company_name: "Tesla", data_as_of: dataAsOf, theme_count: 1, is_stale: false },
  { ticker: "NVDA", company_name: "NVIDIA", data_as_of: dataAsOf, theme_count: 1, is_stale: false },
  { ticker: "AMD", company_name: "Advanced Micro Devices", data_as_of: dataAsOf, theme_count: 0, is_stale: false },
  { ticker: "AAPL", company_name: "Apple", data_as_of: dataAsOf, theme_count: 1, is_stale: false },
  { ticker: "META", company_name: "Meta Platforms", data_as_of: dataAsOf, theme_count: 1, is_stale: false },
];

const story = {
  id: "tsla-1",
  headline: "Tesla delivery coverage remains in focus",
  outlet: "Fixture Markets",
  url: "https://example.com/tsla-deliveries",
  published_at: "2026-07-15T13:15:00Z",
};

function makeThemePayload(overrides = {}) {
  return {
    ticker: "TSLA",
    date: "2026-07-15",
    data_as_of: dataAsOf,
    themes: [
      {
        id: "tsla-deliveries",
        label: "Delivery updates",
        rank: 1,
        sentences: [
          { text: "Coverage today focused on Tesla delivery updates.", citation_ids: ["tsla-1"] },
          { text: "Several outlets discussed the latest reported figures.", citation_ids: ["tsla-1"] },
        ],
        citations: [story],
        stories: [story],
        outlet_count: 1,
        story_count: 1,
        degraded: false,
      },
    ],
    other_coverage: { outlet_count: 0, story_count: 0, stories: [] },
    ...overrides,
  };
}

function jsonResponse(payload, ok = true) {
  return Promise.resolve({
    ok,
    status: ok ? 200 : 500,
    json: async () => payload,
  });
}

function mockApi({ themePayload = makeThemePayload(), status = {}, fail = false } = {}) {
  vi.stubGlobal(
    "fetch",
    vi.fn((url) => {
      if (fail) return jsonResponse({ detail: "Temporary failure" }, false);
      if (url.endsWith("/api/v1/tickers")) return jsonResponse({ data_as_of: dataAsOf, tickers });
      if (url.endsWith("/api/v1/meta/status")) {
        return jsonResponse({ data_as_of: dataAsOf, is_stale: false, last_runs: [], ...status });
      }
      if (url.includes("/api/v1/tickers/") && url.endsWith("/themes")) {
        return jsonResponse(themePayload);
      }
      throw new Error(`Unexpected request: ${url}`);
    }),
  );
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("Ticker Narratives states", () => {
  it("renders the approved loading state while requests are pending", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));

    render(<App />);

    expect(screen.getByText("Loading current coverage…")).toBeInTheDocument();
  });

  it("renders all fixture-backed tickers and cited theme content on success", async () => {
    mockApi();

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Delivery updates" })).toBeInTheDocument();
    expect(screen.getAllByRole("button")).toHaveLength(5);
    expect(screen.getByRole("button", { name: /TSLA/ })).toBeInTheDocument();
    for (const citationLink of screen.getAllByTitle("Open source: Tesla delivery coverage remains in focus")) {
      expect(citationLink).toHaveAttribute("href", story.url);
      expect(citationLink).toHaveAttribute("target", "_blank");
    }
    expect(
      screen.getByText("AI-generated from cited sources. Informational only — not investment advice."),
    ).toBeInTheDocument();
  });

  it("renders the approved error state when an API request fails", async () => {
    mockApi({ fail: true });

    render(<App />);

    expect(
      await screen.findByText("Coverage is temporarily unavailable. Please try again shortly."),
    ).toBeInTheDocument();
  });

  it("renders the approved empty state when a ticker has no coverage", async () => {
    mockApi({ themePayload: makeThemePayload({ themes: [] }) });

    render(<App />);

    expect(await screen.findByText("No current coverage for TSLA.")).toBeInTheDocument();
    expect(screen.getByText("Check back after the next update.")).toBeInTheDocument();
  });

  it("renders the stale banner from API freshness data", async () => {
    mockApi({ status: { is_stale: true } });

    render(<App />);

    expect(
      await screen.findByText(/Data may be delayed\. Last successful update: \d{2}:\d{2}\./),
    ).toBeInTheDocument();
  });

  it("renders the degraded state without an uncited summary", async () => {
    mockApi({
      themePayload: makeThemePayload({
        themes: [
          {
            ...makeThemePayload().themes[0],
            sentences: [],
            degraded: true,
          },
        ],
      }),
    });

    render(<App />);

    expect(
      await screen.findByText("Summary unavailable — source stories are still available"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Coverage today focused on Tesla delivery updates.")).not.toBeInTheDocument();
  });
});
