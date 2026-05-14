import { useState } from "react";

export default function NewsFeed({ news = [], socialPosts = [] }) {
  const [activeTab, setActiveTab] = useState("news");
  const items = activeTab === "news" ? news : socialPosts;

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

      {items.length > 0 ? (
        <div className="feed-list">
          {activeTab === "news"
            ? news.map((item, index) => <NewsItem key={item.id || index} item={item} />)
            : socialPosts.map((post, index) => (
                <SocialPost key={post.id || index} post={post} />
              ))}
        </div>
      ) : (
        <div className="empty-state">
          This API response does not currently include {activeTab === "news" ? "headline" : "social"} items.
        </div>
      )}
    </section>
  );
}

function NewsItem({ item }) {
  const sentiment = item.sentiment?.sentiment_label;

  return (
    <a
      href={item.url || "#"}
      className="news-item"
      target={item.url ? "_blank" : undefined}
      rel={item.url ? "noreferrer" : undefined}
    >
      <div className="news-meta">
        <span className="news-source">{item.source || "Source"}</span>
        <span className="news-time">
          {sentiment ? `${sentiment} · ` : ""}
          {item.time || "N/A"}
        </span>
      </div>
      <p className="news-headline">
        {item.headline || item.title || "No headline available."}
      </p>
    </a>
  );
}

function SocialPost({ post }) {
  return (
    <div className="social-post">
      <div className="social-meta">
        <span className="social-platform">{post.platform || "Social"}</span>
        <span className="social-handle">{post.handle || "Unknown"}</span>
      </div>
      <p className="social-content">{post.content || "No social post content available."}</p>
    </div>
  );
}
