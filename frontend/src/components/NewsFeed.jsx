import { useState } from "react";

export default function NewsFeed({
  news = [],
  socialPosts = [],
  loading = false,
}) {
  const [activeTab, setActiveTab] = useState("news");
  const items = activeTab === "news" ? news : socialPosts;
  const itemLabel = activeTab === "news" ? "headline" : "social post";
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
            Social
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
                <SocialPost key={post.id || index} post={post} />
              ))}
        </div>
      ) : (
          <div className="empty-state">
            {activeTab === "news"
              ? "No headlines available for this ticker yet."
              : "No social posts available for this ticker yet."}
          </div>
      )}
    </section>
  );
}

function NewsItem({ item }) {
  const sentiment = item.sentiment?.sentiment_label || "neutral";
  const confidence = Math.round((item.sentiment?.sentiment_confidence || 0) * 100);

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
        {item.time ? <span className="news-time">{item.time}</span> : null}
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

function SocialPost({ post }) {
  const sentiment = post.sentiment?.sentiment_label || "neutral";
  const confidence = Math.round((post.sentiment?.sentiment_confidence || 0) * 100);

  return (
    <div className="social-post">
      <div className="social-meta">
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
