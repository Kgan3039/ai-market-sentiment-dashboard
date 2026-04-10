import { useState } from "react";

export default function NewsFeed({ news, socialPosts }) {
  const [activeTab, setActiveTab] = useState("news");

  return (
    <section className="card">
      <div className="section-header">
        <h2 className="section-title">Market Pulse</h2>
        <div className="toggle-group">
          <button className={`toggle-btn ${activeTab === "news" ? "active" : ""}`} onClick={() => setActiveTab("news")}>
            Headlines
          </button>
          <button className={`toggle-btn ${activeTab === "social" ? "active" : ""}`} onClick={() => setActiveTab("social")}>
            Social
          </button>
        </div>
      </div>

      {/* Conditional rendering — show one tab's content based on state */}
      {activeTab === "news" && (
        <div className="feed-list">
          {/* .map() loops over the array and renders a component for each item */}
          {news.map((item) => (
            <NewsItem key={item.id} item={item} />
          ))}
        </div>
      )}

      {activeTab === "social" && (
        <div className="feed-list">
          {socialPosts.map((post) => (
            <SocialPost key={post.id} post={post} />
          ))}
        </div>
      )}
    </section>
  );
}

function NewsItem({ item }) {
  return (
    <a href={item.url} className="news-item">
      <div className="news-meta">
        <span className="news-source">{item.source}</span>
        <span className="news-time">{item.time}</span>
        <span className={`sentiment-pill ${item.sentiment}`}>{item.sentiment}</span>
      </div>
      <p className="news-headline">{item.headline}</p>
    </a>
  );
}

function SocialPost({ post }) {
  return (
    <div className="social-post">
      <div className="social-meta">
        <span className="social-platform">{post.platform}</span>
        <span className="social-handle">{post.handle}</span>
        <span className={`sentiment-pill ${post.sentiment}`}>{post.sentiment}</span>
        <span className="social-likes">♥ {post.likes}</span>
      </div>
      <p className="social-content">"{post.content}"</p>
    </div>
  );
}
