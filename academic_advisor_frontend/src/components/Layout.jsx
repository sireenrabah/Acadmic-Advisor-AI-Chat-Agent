// src/components/Layout.jsx
import { Outlet } from "react-router-dom";
import Header from "./Header";

export default function Layout() {
  return (
    <>
      <Header />             {/* fixed, green, always visible */}
      <div className="page-offset">
        <Outlet />           {/* Start / Setup / Chat render here */}
      </div>
    </>
  );
}
