import { useState } from "react";

export default function NewsFeed({
  news = [],
  socialPosts = [],
  socialAvailability,
  loading = false,
}) {
  const [activeTab, setActiveTab] = useState("news");
  const socialSummary = getPulseItemSummary(socialPosts, socialAvailability);
  const items = activeTab === "news" ? news : socialPosts;
  const itemLabel = activeTab === "news" ? "headline" : socialSummary.itemLabel;
  const countLabel = `${items.length} ${itemLabel}${items.length === 1 ? "" : "s"} available.`;

  return (
    <section className="card">
      <div className="section-header">
        <h2 className="section-title">Market Pulse</h2>
        <div className="toggle-group">
          <button
            className={`toggle-btn ${activeTab === "news" ? "active" : ""}`}
            onClick={() => setActiveTab("news")}
          >
            Headlines
          </button>
          <button
            className={`toggle-btn ${activeTab === "social" ? "active" : ""}`}
            onClick={() => setActiveTab("social")}
          >
            {socialSummary.tabLabel}
          </button>
        </div>
      </div>

      <div className="feed-summary">
        {loading
          ? "Fetching market pulse items."
          : countLabel}
      </div>

      {loading ? (
        <div className="empty-state">Loading {itemLabel} items...</div>
      ) : items.length > 0 ? (
        <div className="feed-list">
          {activeTab === "news"
            ? news.map((item, index) => <NewsItem key={item.id || index} item={item} />)
            : socialPosts.map((post, index) => (
                <PulsePost
                  key={post.id || index}
                  post={post}
                  fallbackLabel={socialSummary.itemTitle}
                />
              ))}
        </div>
      ) : (
          <div className="empty-state">
            {activeTab === "news"
              ? "No headlines available for this ticker yet."
              : socialSummary.emptyMessage}
          </div>
      )}
    </section>
  );
}

function getPulseItemSummary(posts, availability) {
  const publisherCount = posts.filter((post) => post.content_type !== "social_post").length;
  const socialCount = posts.length - publisherCount;
  const backendSource = availability?.source || "";
  const publisherOnly =
    posts.length > 0 &&
    publisherCount === posts.length &&
    (backendSource.toLowerCase().includes("publisher") || socialCount === 0);

  if (publisherOnly) {
    return {
      tabLabel: "Publisher Items",
      itemLabel: "publisher item",
      itemTitle: "Publisher item",
      emptyMessage: "No publisher fallback items available for this ticker yet.",
    };
  }

  if (publisherCount > 0 && socialCount > 0) {
    return {
      tabLabel: "Mixed Sources",
      itemLabel: "market pulse item",
      itemTitle: "Market pulse item",
      emptyMessage: "No mixed market pulse items available for this ticker yet.",
    };
  }

  return {
    tabLabel: "Social",
    itemLabel: "social post",
    itemTitle: "Social post",
    emptyMessage: "No social posts available for this ticker yet.",
  };
}

function formatHeadlineTime(item) {
  if (item.time) return item.time;
  if (!item.published_at) return null;

  const parsed = new Date(item.published_at);
  if (Number.isNaN(parsed.getTime())) return null;

  return parsed.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function NewsItem({ item }) {
  const sentiment = item.sentiment?.sentiment_label || "neutral";
  const confidence = Math.round((item.sentiment?.sentiment_confidence || 0) * 100);
  const displayTime = formatHeadlineTime(item);

  return (
    <a
      href={item.url || "#"}
      className="news-item"
      target={item.url ? "_blank" : undefined}
      rel={item.url ? "noreferrer" : undefined}
    >
      <div className="news-meta">
        <span className="news-source">{item.source || "Source"}</span>
        <span className={`sentiment-pill ${sentiment}`}>{sentiment}</span>
        {displayTime ? <span className="news-time">{displayTime}</span> : null}
      </div>
      <p className="news-headline">
        {item.headline || item.title || "No headline available."}
      </p>
      {item.summary ? <p className="news-summary">{item.summary}</p> : null}
      {confidence > 0 ? (
        <div className="news-footer">Sentiment confidence {confidence}%</div>
      ) : null}
    </a>
  );
}

function PulsePost({ post, fallbackLabel }) {
  const sentiment = post.sentiment?.sentiment_label || "neutral";
  const confidence = Math.round((post.sentiment?.sentiment_confidence || 0) * 100);
  const itemLabel =
    post.content_type === "social_post" ? "Social post" : fallbackLabel || "Publisher item";

  return (
    <div className="social-post">
      <div className="social-meta">
        <span className="social-platform">{itemLabel}</span>
        <span className="social-platform">{post.source || "Source"}</span>
        <span className={`sentiment-pill ${sentiment}`}>{sentiment}</span>
        {post.date ? <span className="social-date">{post.date}</span> : null}
      </div>
      <p className="social-content">{post.text || "No social post content available."}</p>
      {confidence > 0 ? (
        <div className="news-footer">Sentiment confidence {confidence}%</div>
      ) : null}
    </div>
  );
}
