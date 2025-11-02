// src/components/Header.jsx
import logo from "../assets/logo.png"; 

export default function Header() {
  return (
    <header className="site-header fixed">
      <div className="container header-inner">
        <div className="brand">
          <img src={logo} alt="AI Academic Advisor" className="brand-logo" />
          <span className="brand-name">AI Academic Advisor</span>
        </div>
        {/* no nav tabs */}
      </div>
    </header>
  );
}
