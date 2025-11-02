import React from "react";

export default class ErrorBoundary extends React.Component {
  constructor(props){ super(props); this.state = { hasError:false, error:null }; }
  static getDerivedStateFromError(error){ return { hasError:true, error }; }
  componentDidCatch(err, info){ console.error("ErrorBoundary caught:", err, info); }
  render(){
    if (this.state.hasError) {
      return <pre style={{padding:16,color:"#b91c1c",whiteSpace:"pre-wrap"}}>
        {String(this.state.error)}
      </pre>;
    }
    return this.props.children;
  }
}
