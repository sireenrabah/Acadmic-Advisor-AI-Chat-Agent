import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import "../styles/theme.css";
import "../styles/results.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

function toPercent(score) {
  if (typeof score !== "number" || Number.isNaN(score)) return null;
  const val = score <= 1 ? score * 100 : score;
  return Math.max(0, Math.min(100, Math.round(val)));
}

export default function ResultsPage() {
  const nav = useNavigate();
  const { id } = useParams();

  const [summaryData, setSummaryData] = useState(null);
  const [majorDetails, setMajorDetails] = useState({});
  const [loading, setLoading] = useState(true);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    // Load summary from localStorage (stored by ChatPage)
    const stored = localStorage.getItem("aa_final_summary");
    if (stored) {
      try {
        const data = JSON.parse(stored);
        setSummaryData(data);
        
        // Fetch details for each major
        if (data.recommendedMajors && data.recommendedMajors.length > 0) {
          setLoadingDetails(true);
          fetchMajorDetails(data.sessionId, data.recommendedMajors, data.language);
        }
      } catch (e) {
        console.error("Failed to parse summary:", e);
        setError("Failed to load summary data");
      }
    } else {
      setError("No summary data available. Please complete a chat session first.");
    }
    setLoading(false);
  }, [id]);

  const fetchMajorDetails = async (sessionId, majors, lang) => {
    const details = {};
    
    // First, we need to ensure the session has the recommended majors stored
    // Send them to the backend
    try {
      await fetch(`${API_BASE}/recommendations?session_id=${sessionId}`, {
        method: "GET"
      });
    } catch (e) {
      console.error("Failed to ensure session has recommendations:", e);
    }
    
    for (let idx = 0; idx < majors.length; idx++) {
      const major = majors[idx];
      console.log(`Fetching details for major ${idx + 1}: ${major.name}`);
      
      try {
        // NEW: Use major_index parameter for stateless detail fetching
        // This prevents session state corruption when fetching all majors at once
        console.log(`  Fetching details for major index ${idx} (stateless)`);
        
        const [resp1, resp2, resp3, resp4] = await Promise.all([
          fetch(`${API_BASE}/turn/followup`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              session_id: sessionId,
              option: "1",
              majors: majors,
              major_index: idx  // NEW: Stateless major selection
            })
          }),
          fetch(`${API_BASE}/turn/followup`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              session_id: sessionId,
              option: "2",
              majors: majors,
              major_index: idx  // NEW: Stateless major selection
            })
          }),
          fetch(`${API_BASE}/turn/followup`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              session_id: sessionId,
              option: "3",
              majors: majors,
              major_index: idx  // NEW: Stateless major selection
            })
          }),
          fetch(`${API_BASE}/turn/followup`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              session_id: sessionId,
              option: "4",
              majors: majors,
              major_index: idx  // NEW: Stateless major selection
            })
          })
        ]);

        const [data1, data2, data3, data4] = await Promise.all([
          resp1.json(),
          resp2.json(),
          resp3.json(),
          resp4.json()
        ]);
        
        console.log(`  Received details:`, { data1, data2, data3, data4 });

        // CRITICAL: Store by index, not by name (names might not match exactly)
        const majorKey = `major_${idx}`;  // Use index as key to avoid name mismatch
        details[majorKey] = {
          whyGood: data1.text || data1.message || "Information unavailable",
          eligibility: data2.text || data2.message || "Information unavailable",
          courses: data3.text || data3.message || "Information unavailable",
          career: data4.text || data4.message || "Information unavailable",
          majorName: major.name  // Store the display name separately
        };
        
        console.log(`  Stored details for index ${idx} (${major.name}):`, details[majorKey]);
      } catch (e) {
        console.error(`Failed to fetch details for ${major.name}:`, e);
        const majorKey = `major_${idx}`;
        details[majorKey] = {
          whyGood: "Information unavailable",
          eligibility: "Information unavailable",
          courses: "Information unavailable",
          career: "Information unavailable",
          majorName: major.name
        };
      }
    }
    
    console.log("All details fetched:", details);
    setMajorDetails(details);
    setLoadingDetails(false);
  };

  const downloadSummary = async () => {
    if (!summaryData) return;

    const { recommendedMajors, language, degree } = summaryData;
    const isRTL = language === 'he' || language === 'ar';
    
    // For Hebrew/Arabic, use html2canvas to capture the page as image
    if (isRTL) {
      try {
        // Dynamically import html2canvas and jsPDF
        const html2canvas = (await import('html2canvas')).default;
        const { jsPDF } = await import('jspdf');
        
        // Find the results content area
        const resultsElement = document.querySelector('.grid');
        if (!resultsElement) {
          alert('Could not find content to export');
          return;
        }
        
        // Create a temporary container for better PDF rendering
        const tempContainer = document.createElement('div');
        tempContainer.style.cssText = `
          position: absolute;
          left: -9999px;
          width: 800px;
          padding: 40px;
          background: white;
          direction: ${isRTL ? 'rtl' : 'ltr'};
        `;
        
        // Create header
        const header = document.createElement('div');
        header.style.cssText = 'margin-bottom: 30px; text-align: center;';
        header.innerHTML = `
          <h1 style="font-size: 24px; margin: 0 0 10px 0; color: #1f2937;">
            ${language === 'he' ? 'יעוץ אקדמי - סיכום' : 'الاستشارة الأكاديمية - ملخص'}
          </h1>
          <p style="font-size: 14px; color: #6b7280; margin: 5px 0;">
            ${language === 'he' ? 'תואר' : 'درجة'}: ${degree}
          </p>
          <p style="font-size: 14px; color: #6b7280; margin: 5px 0;">
            ${language === 'he' ? 'תאריך' : 'تاريخ'}: ${new Date().toLocaleDateString()}
          </p>
          <hr style="margin: 20px 0; border: none; border-top: 2px solid #e5e7eb;" />
        `;
        
        // Clone and append content
        const contentClone = resultsElement.cloneNode(true);
        tempContainer.appendChild(header);
        tempContainer.appendChild(contentClone);
        document.body.appendChild(tempContainer);
        
        // Generate canvas from the content
        const canvas = await html2canvas(tempContainer, {
          scale: 2,
          useCORS: true,
          backgroundColor: '#ffffff',
          windowWidth: 800
        });
        
        // Remove temp container
        document.body.removeChild(tempContainer);
        
        // Create PDF
        const imgData = canvas.toDataURL('image/png');
        const pdf = new jsPDF({
          orientation: 'portrait',
          unit: 'mm',
          format: 'a4'
        });
        
        const imgWidth = 210; // A4 width in mm
        const imgHeight = (canvas.height * imgWidth) / canvas.width;
        const pageHeight = 297; // A4 height in mm
        let heightLeft = imgHeight;
        let position = 0;
        
        // Add image to PDF (with pagination if needed)
        pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
        heightLeft -= pageHeight;
        
        while (heightLeft > 0) {
          position = heightLeft - imgHeight;
          pdf.addPage();
          pdf.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
          heightLeft -= pageHeight;
        }
        
        // Save with language-appropriate filename
        const dateStr = new Date().toISOString().slice(0,10);
        const filename = language === 'he' ? `יעוץ-אקדמי-${dateStr}.pdf` :
                         language === 'ar' ? `استشارة-أكاديمية-${dateStr}.pdf` :
                         `academic-advisor-summary-${dateStr}.pdf`;
        pdf.save(filename);
        return;
      } catch (error) {
        console.error('Error generating PDF:', error);
        alert('Failed to generate PDF. Please try again.');
        return;
      }
    }
    
    // For English, use the original jsPDF text method
    const { jsPDF } = await import('jspdf');
    const doc = new jsPDF();
    let y = 20;
    
    // Title
    doc.setFontSize(18);
    doc.setFont('helvetica', 'bold');
    doc.text('ACADEMIC ADVISOR - SUMMARY', 105, y, { align: 'center' });
    y += 15;
    
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text(`Degree: ${degree}`, 20, y);
    y += 6;
    doc.text(`Date: ${new Date().toLocaleDateString()}`, 20, y);
    y += 10;
    
    doc.setLineWidth(0.5);
    doc.line(20, y, 190, y);
    y += 10;
    
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('RECOMMENDED MAJORS', 20, y);
    y += 10;
    
    recommendedMajors.forEach((major, idx) => {
      const pct = toPercent(major.score);
      // FIXED: Use index-based key to avoid name mismatch
      const majorKey = `major_${idx}`;
      const details = majorDetails[majorKey] || {};
      
      // Check if we need a new page
      if (y > 240) {
        doc.addPage();
        y = 20;
      }
      
      // Major name and number
      doc.setFontSize(14);
      doc.setFont('helvetica', 'bold');
      doc.text(`${idx + 1}. ${major.name}`, 20, y);
      y += 8;
      
      // Match score
      doc.setFontSize(11);
      doc.setFont('helvetica', 'bold');
      doc.text(`Match: ${pct}%`, 25, y);
      y += 7;
      
      // Rationale
      if (major.rationale) {
        doc.setFontSize(10);
        doc.setFont('helvetica', 'normal');
        const rationaleLines = doc.splitTextToSize(major.rationale, 160);
        doc.text(rationaleLines, 25, y);
        y += (rationaleLines.length * 5) + 5;
      }
      
      // Helper to add sections with bullet points
      const addSection = (emoji, title, content, maxChars = 150) => {
        const summarized = summarizeText(content, maxChars);
        if (!summarized) return;
        
        if (y > 250) {
          doc.addPage();
          y = 20;
        }
        
        doc.setFontSize(10);
        doc.setFont('helvetica', 'bold');
        doc.text(`${emoji} ${title}`, 25, y);
        y += 6;
        
        doc.setFontSize(9);
        doc.setFont('helvetica', 'normal');
        
        const sentences = summarized
          .split(/[.!?]\s+/)
          .filter(s => s.trim().length > 10)
          .slice(0, 4);
        
        sentences.forEach(sentence => {
          if (y > 275) {
            doc.addPage();
            y = 20;
          }
          const bulletLines = doc.splitTextToSize(`• ${sentence.trim()}.`, 155);
          doc.text(bulletLines, 30, y);
          y += (bulletLines.length * 4.5) + 2;
        });
        
        y += 3;
      };
      
      // Add all detail sections
      addSection('💡', 'Key Points', details.whyGood, 150);
      addSection('📋', 'Main Requirements', details.eligibility, 150);
      addSection('📚', 'Core Courses', details.courses, 120);
      addSection('🎯', 'Career Options', details.career, 120);
      
      // Space between majors
      y += 8;
    });
    
    // Add footer note
    if (y > 270) {
      doc.addPage();
      y = 20;
    }
    
    doc.setFontSize(9);
    doc.setFont('helvetica', 'italic');
    const footerText = 'This report was generated by Ramat Gan College Academic Advisor';
    const footerLines = doc.splitTextToSize(footerText, 170);
    doc.text(footerLines, 20, y);
    
    // Save
    doc.save(`academic-advisor-summary-${new Date().toISOString().slice(0,10)}.pdf`);
  };

  if (loading) {
    return (
      <div className="results-wrap">
        <div className="results container">
          <div className="card loader">Loading summary...</div>
        </div>
      </div>
    );
  }

  if (error || !summaryData) {
    return (
      <div className="results-wrap">
        <div className="results container">
          <div className="card error">{error || "No data available"}</div>
          <div className="foot-actions">
            <button className="btn cta" onClick={() => nav("/")}>Start New Interview</button>
          </div>
        </div>
      </div>
    );
  }

  const { recommendedMajors, language } = summaryData;
  const isRTL = language === 'he' || language === 'ar';

  // Helper to clean backend response text (remove emoji headers and markdown)
  const cleanDetailText = (text) => {
    if (!text || text === "Information unavailable") return '';
    
    // Remove emoji headers like "💡 **title**" or "📋 **title - major name**"
    let cleaned = text.replace(/^[💡📋📚🎯✅⚠️]\s*\*\*[^*]+\*\*\s*/gm, '');
    
    // Remove separator lines
    cleaned = cleaned.replace(/^─+$/gm, '');
    
    // Remove extra markdown bold markers
    cleaned = cleaned.replace(/\*\*/g, '');
    
    // Remove leading bullet markers (• or numbers)
    cleaned = cleaned.replace(/^[•\d]+\.\s*/gm, '');
    
    // Trim whitespace and multiple newlines
    cleaned = cleaned.trim().replace(/\n\n+/g, '\n');
    
    return cleaned;
  };

  // Helper function to summarize long text (keep only key points)
  const summarizeText = (text, maxLength = 200) => {
    if (!text || text === "Information unavailable") return null;
    if (text.length <= maxLength) return text;
    
    // Split by sentences or lines
    const sentences = text.split(/[.!?\n]+/).filter(s => s.trim());
    let summary = '';
    
    for (const sentence of sentences) {
      if ((summary + sentence).length > maxLength) break;
      summary += sentence.trim() + '. ';
    }
    
    return summary.trim() || text.substring(0, maxLength) + '...';
  };

  return (
    <div className="results-wrap">
      <div className="results container" style={{ display: 'flex', gap: 24, alignItems: 'flex-start' }}>
        
        {/* LEFT SIDEBAR - Action Buttons */}
        <div style={{ 
          width: 220, 
          flexShrink: 0,
          position: 'sticky',
          top: 20
        }}>
          <div className="card" style={{ padding: 20 }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16, textAlign: 'center' }}>
              {language === 'he' ? 'פעולות' : 
               language === 'ar' ? 'الإجراءات' :
               'Actions'}
            </h3>
            
            <button 
              className="btn cta"
              onClick={downloadSummary}
              style={{ 
                width: '100%', 
                marginBottom: 12,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' 
              }}
            >
              {language === 'he' ? '📥 הורד סיכום' : 
               language === 'ar' ? '📥 تحميل الملخص' :
               '📥 Download Summary'}
            </button>
            
            <button 
              className="btn outline"
              onClick={() => nav(`/chat/${id}`)}
              style={{ width: '100%', marginBottom: 12 }}
            >
              {language === 'he' ? '↩️ חזור לשיחה' :
               language === 'ar' ? '↩️ العودة إلى المحادثة' :
               '↩️ Back to Chat'}
            </button>
            
            <button 
              className="btn outline"
              onClick={() => nav("/")}
              style={{ width: '100%' }}
            >
              {language === 'he' ? '🆕 ראיון חדש' :
               language === 'ar' ? '🆕 مقابلة جديدة' :
               '🆕 New Interview'}
            </button>
          </div>
        </div>

        {/* RIGHT CONTENT - Summary */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="results-header" style={{ textAlign: isRTL ? 'right' : 'left', direction: isRTL ? 'rtl' : 'ltr' }}>
            <h1 className="page-title accent" style={{ textAlign: isRTL ? 'right' : 'left' }}>
              {language === 'he' ? 'המלצות סופיות' : 
               language === 'ar' ? 'التوصيات النهائية' :
               'Final Recommendations'}
            </h1>
            <p className="muted" style={{ textAlign: isRTL ? 'right' : 'left' }}>
              {language === 'he' ? 'על בסיס השיחה והציונים שלך, הנה המגמות המומלצות' :
               language === 'ar' ? 'بناءً على محادثتك ودرجاتك، إليك التخصصات الموصى بها' :
               'Based on your conversation and grades, here are your recommended majors'}
            </p>
          </div>

        {loadingDetails && (
          <div className="card" style={{ textAlign: 'center', padding: 24, marginBottom: 24 }}>
            <div className="loader" style={{ fontSize: 14 }}>
              {language === 'he' ? '⏳ טוען פרטים מלאים...' :
               language === 'ar' ? '⏳ تحميل التفاصيل الكاملة...' :
               '⏳ Loading full details...'}
            </div>
          </div>
        )}

        {recommendedMajors.length > 0 && (
          <div className="grid">
            {recommendedMajors.map((major, idx) => {
              const pct = toPercent(major.score);
              // FIXED: Use index-based key to avoid name mismatch
              const majorKey = `major_${idx}`;
              const details = majorDetails[majorKey] || {};

              return (
                <div key={idx} className="rec-card card" style={{ 
                  marginBottom: 24, 
                  direction: isRTL ? 'rtl' : 'ltr', 
                  textAlign: isRTL ? 'right' : 'left' 
                }}>
                  <div className="rec-head">
                    <span className="badge-rank">#{idx + 1}</span>
                    <h2 className="rec-title" style={{ textAlign: isRTL ? 'right' : 'left' }}>
                      {major.name || major.english_name || 'Major'}
                    </h2>
                  </div>

                  {pct != null && (
                    <div className="scorerow" aria-label={`Match score ${pct}%`}>
                      <span className="scorelabel">
                        {language === 'he' ? 'התאמה' : language === 'ar' ? 'المطابقة' : 'Match'}
                      </span>
                      <div className="scorebar">
                        <div className="fill" style={{ width: `${pct}%` }} />
                      </div>
                      <span className="scorepct">{pct}%</span>
                    </div>
                  )}

                  {major.rationale && (
                    <p className="muted" style={{ 
                      marginTop: 12, 
                      padding: 12, 
                      background: '#f9fafb', 
                      borderRadius: 8,
                      textAlign: isRTL ? 'right' : 'left',
                      lineHeight: 1.6
                    }}>
                      {major.rationale}
                    </p>
                  )}

                  {/* Detailed sections with bullet points */}
                  {details.whyGood && (
                    <div style={{ marginTop: 16, textAlign: isRTL ? 'right' : 'left' }}>
                      <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 8, color: '#667eea' }}>
                        💡 {language === 'he' ? 'נקודות מפתח' : language === 'ar' ? 'نقاط رئيسية' : 'Key Points'}
                      </h3>
                      <ul style={{ 
                        fontSize: 13, 
                        lineHeight: 1.8, 
                        margin: 0, 
                        paddingInlineStart: isRTL ? 0 : 20,
                        paddingInlineEnd: isRTL ? 20 : 0,
                        listStylePosition: 'inside'
                      }}>
                        {cleanDetailText(details.whyGood).split(/[.!]\s+/).filter(s => s.trim().length > 15).slice(0, 4).map((point, i) => (
                          <li key={i} style={{ marginBottom: 6 }}>{point.trim()}.</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {details.eligibility && (
                    <div style={{ marginTop: 16, textAlign: isRTL ? 'right' : 'left' }}>
                      <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 8, color: '#10b981' }}>
                        📋 {language === 'he' ? 'דרישות עיקריות' : language === 'ar' ? 'المتطلبات الرئيسية' : 'Main Requirements'}
                      </h3>
                      <ul style={{ 
                        fontSize: 13, 
                        lineHeight: 1.8, 
                        margin: 0, 
                        paddingInlineStart: isRTL ? 0 : 20,
                        paddingInlineEnd: isRTL ? 20 : 0,
                        listStylePosition: 'inside'
                      }}>
                        {cleanDetailText(details.eligibility).split(/[.!]\s+/).filter(s => s.trim().length > 15).slice(0, 4).map((point, i) => (
                          <li key={i} style={{ marginBottom: 6 }}>{point.trim()}.</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {details.courses && (
                    <div style={{ marginTop: 16, textAlign: isRTL ? 'right' : 'left' }}>
                      <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 8, color: '#f59e0b' }}>
                        📚 {language === 'he' ? 'קורסים מרכזיים' : language === 'ar' ? 'دورات رئيسية' : 'Core Courses'}
                      </h3>
                      <ul style={{ 
                        fontSize: 13, 
                        lineHeight: 1.8, 
                        margin: 0, 
                        paddingInlineStart: isRTL ? 0 : 20,
                        paddingInlineEnd: isRTL ? 20 : 0,
                        listStylePosition: 'inside'
                      }}>
                        {cleanDetailText(details.courses).split(/[.!]\s+/).filter(s => s.trim().length > 15).slice(0, 5).map((point, i) => (
                          <li key={i} style={{ marginBottom: 6 }}>{point.trim()}.</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {details.career && (
                    <div style={{ marginTop: 16, textAlign: isRTL ? 'right' : 'left' }}>
                      <h3 style={{ fontSize: 14, fontWeight: 700, marginBottom: 8, color: '#ef4444' }}>
                        🎯 {language === 'he' ? 'אפשרויות קריירה' : language === 'ar' ? 'خيارات مهنية' : 'Career Options'}
                      </h3>
                      <ul style={{ 
                        fontSize: 13, 
                        lineHeight: 1.8, 
                        margin: 0, 
                        paddingInlineStart: isRTL ? 0 : 20,
                        paddingInlineEnd: isRTL ? 20 : 0,
                        listStylePosition: 'inside'
                      }}>
                        {cleanDetailText(details.career).split(/[.!]\s+/).filter(s => s.trim().length > 15).slice(0, 5).map((point, i) => (
                          <li key={i} style={{ marginBottom: 6 }}>{point.trim()}.</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
        </div>
      </div>
    </div>
  );
}
