"""
AI Trading Copilot - PDF Analysis Service
Extracts text from PDFs and provides AI-powered analysis.
Supports both text-based and scanned/image-based PDFs via OCR.
"""
import io
import re
from typing import Optional, Dict, Any, List
from datetime import datetime

try:
    import pdfplumber
    PDF_PLUMBER_AVAILABLE = True
except ImportError:
    PDF_PLUMBER_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

# OCR support
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class PDFService:
    """Service for extracting and analyzing PDF content."""
    
    def __init__(self):
        self.llm = None
        self._init_llm()
        # Store document context for Q&A
        self._document_cache: Dict[str, Dict[str, Any]] = {}
    
    def _init_llm(self):
        """Initialize LLM for analysis."""
        try:
            from config import settings
            if settings.groq_api_key:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(
                    api_key=settings.groq_api_key,
                    model_name=settings.model_name,
                    temperature=0.3,
                )
                print("✅ PDF Service: Groq LLM initialized")
        except Exception as e:
            print(f"⚠️ PDF Service: LLM initialization failed: {e}")
            self.llm = None
    
    def _extract_text_with_ocr(self, pdf_content: bytes) -> str:
        """Extract text from PDF using OCR (for scanned documents)."""
        text = ""
        
        if not PYMUPDF_AVAILABLE:
            print("PyMuPDF not available for OCR")
            return ""
        
        if not TESSERACT_AVAILABLE:
            print("Tesseract not available for OCR")
            return ""
        
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Convert page to image (high resolution for better OCR)
                mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # Run OCR
                page_text = pytesseract.image_to_string(image, lang='eng')
                if page_text:
                    text += page_text + "\n\n"
            
            pdf_document.close()
            return text.strip()
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes. Tries multiple methods."""
        text = ""
        
        # Try PyMuPDF first (often best for complex PDFs)
        if PYMUPDF_AVAILABLE:
            try:
                pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    page_text = page.get_text("text")
                    if page_text:
                        text += page_text + "\n\n"
                pdf_document.close()
                
                if text.strip():
                    print(f"✅ Extracted {len(text)} chars with PyMuPDF")
                    return text.strip()
            except Exception as e:
                print(f"PyMuPDF text extraction failed: {e}")
        
        # Try pdfplumber (better for tables)
        if PDF_PLUMBER_AVAILABLE:
            try:
                with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                        
                        # Extract tables if any
                        tables = page.extract_tables()
                        for table in tables:
                            for row in table:
                                if row:
                                    text += " | ".join([str(cell) if cell else "" for cell in row]) + "\n"
                            text += "\n"
                
                if text.strip():
                    print(f"✅ Extracted {len(text)} chars with pdfplumber")
                    return text.strip()
            except Exception as e:
                print(f"pdfplumber failed: {e}")
        
        # Fallback to PyPDF2
        if PYPDF2_AVAILABLE:
            try:
                reader = PdfReader(io.BytesIO(pdf_content))
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                
                if text.strip():
                    print(f"✅ Extracted {len(text)} chars with PyPDF2")
                    return text.strip()
            except Exception as e:
                print(f"PyPDF2 failed: {e}")
        
        # Last resort: OCR for scanned documents
        print("Text extraction failed, trying OCR...")
        ocr_text = self._extract_text_with_ocr(pdf_content)
        if ocr_text:
            print(f"✅ Extracted {len(ocr_text)} chars with OCR")
            return ocr_text
        
        return text.strip()
    
    def detect_document_type(self, text: str) -> str:
        """Detect the type of financial document."""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ["quarterly results", "q1", "q2", "q3", "q4", "earnings"]):
            return "earnings_report"
        elif any(kw in text_lower for kw in ["annual report", "annual review", "fiscal year"]):
            return "annual_report"
        elif any(kw in text_lower for kw in ["balance sheet", "assets", "liabilities", "equity"]):
            return "balance_sheet"
        elif any(kw in text_lower for kw in ["income statement", "profit and loss", "p&l", "revenue"]):
            return "income_statement"
        elif any(kw in text_lower for kw in ["cash flow", "operating activities", "investing activities"]):
            return "cash_flow"
        elif any(kw in text_lower for kw in ["portfolio", "holdings", "investment summary"]):
            return "portfolio_statement"
        elif any(kw in text_lower for kw in ["research report", "buy rating", "sell rating", "target price"]):
            return "research_report"
        elif any(kw in text_lower for kw in ["mutual fund", "nav", "scheme"]):
            return "mutual_fund_statement"
        elif any(kw in text_lower for kw in ["demat", "depository", "isin"]):
            return "demat_statement"
        else:
            return "general_financial"
    
    def extract_financial_metrics(self, text: str) -> Dict[str, Any]:
        """Extract key financial metrics from text."""
        metrics = {}
        
        # Common financial patterns
        patterns = {
            "revenue": r"(?:revenue|total\s+income|net\s+sales)[:\s]*(?:rs\.?|₹|inr)?\s*([\d,]+(?:\.\d+)?)\s*(?:cr|crore|lakh|million|billion)?",
            "profit": r"(?:net\s+profit|pat|profit\s+after\s+tax)[:\s]*(?:rs\.?|₹|inr)?\s*([\d,]+(?:\.\d+)?)\s*(?:cr|crore|lakh|million)?",
            "eps": r"(?:eps|earnings\s+per\s+share)[:\s]*(?:rs\.?|₹|inr)?\s*([\d,]+(?:\.\d+)?)",
            "pe_ratio": r"(?:p/e|pe\s+ratio|price[/-]earnings)[:\s]*([\d,]+(?:\.\d+)?)",
            "market_cap": r"(?:market\s+cap(?:italization)?)[:\s]*(?:rs\.?|₹|inr)?\s*([\d,]+(?:\.\d+)?)\s*(?:cr|crore|lakh|million|billion)?",
            "dividend": r"(?:dividend)[:\s]*(?:rs\.?|₹|inr)?\s*([\d,]+(?:\.\d+)?)",
            "growth": r"(?:growth|yoy|y-o-y)[:\s]*([\d,]+(?:\.\d+)?)\s*%",
        }
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, text.lower())
            if matches:
                try:
                    value = matches[0].replace(",", "")
                    metrics[key] = float(value)
                except:
                    pass
        
        return metrics
    
    async def analyze_pdf(self, pdf_content: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
        """Analyze PDF content and provide insights."""
        # Extract text
        text = self.extract_text_from_pdf(pdf_content)
        
        if not text:
            # Check why extraction failed
            if not TESSERACT_AVAILABLE:
                error_msg = "Could not extract text from the PDF. This appears to be a scanned/image-based PDF. To enable OCR support, please install Tesseract OCR: Download from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH."
            else:
                error_msg = "Could not extract text from the PDF. The file might be corrupted or have an unsupported format."
            
            return {
                "success": False,
                "error": error_msg,
                "filename": filename
            }
        
        # Detect document type
        doc_type = self.detect_document_type(text)
        
        # Extract metrics
        metrics = self.extract_financial_metrics(text)
        
        # Truncate text for LLM (max 15000 chars to fit in context)
        truncated_text = text[:15000] if len(text) > 15000 else text
        
        # Generate AI analysis
        analysis = await self._generate_ai_analysis(truncated_text, doc_type, metrics)
        
        # Generate a document ID and store context for Q&A
        import hashlib
        doc_id = hashlib.md5(f"{filename}{text[:100]}".encode()).hexdigest()[:12]
        self._document_cache[doc_id] = {
            "filename": filename,
            "text": truncated_text,
            "doc_type": doc_type,
            "metrics": metrics,
            "analysis": analysis,
            "timestamp": datetime.now()
        }
        
        return {
            "success": True,
            "filename": filename,
            "document_id": doc_id,  # Add document ID for Q&A
            "document_type": doc_type,
            "page_count": text.count("\n\n") // 2 + 1,  # Rough estimate
            "extracted_metrics": metrics,
            "analysis": analysis,
            "text_preview": text[:500] + "..." if len(text) > 500 else text,
            "timestamp": datetime.now().isoformat()
        }
    
    async def ask_question(self, document_id: str, question: str) -> Dict[str, Any]:
        """Answer a question about a previously analyzed document."""
        if document_id not in self._document_cache:
            return {
                "success": False,
                "error": "Document not found. Please analyze the document again."
            }
        
        doc_context = self._document_cache[document_id]
        
        if not self.llm:
            return {
                "success": False,
                "error": "AI service not available"
            }
        
        doc_type_labels = {
            "earnings_report": "Quarterly/Annual Earnings Report",
            "annual_report": "Annual Report",
            "balance_sheet": "Balance Sheet",
            "income_statement": "Income Statement / P&L",
            "cash_flow": "Cash Flow Statement",
            "portfolio_statement": "Portfolio/Investment Statement",
            "research_report": "Stock Research Report",
            "mutual_fund_statement": "Mutual Fund Statement",
            "demat_statement": "Demat Account Statement",
            "general_financial": "Financial Document"
        }
        
        prompt = f"""You are a financial analyst assistant. Answer the user's question based ONLY on the document content provided below.

Document Type: {doc_type_labels.get(doc_context['doc_type'], 'Financial Document')}
Document Name: {doc_context['filename']}

Document Content:
{doc_context['text']}

Extracted Metrics: {doc_context['metrics'] if doc_context['metrics'] else 'None found'}

User Question: {question}

Instructions:
- Answer based only on information in the document
- If the information is not in the document, say so clearly
- Be concise and specific
- Use bullet points for clarity when appropriate
- Reference specific figures from the document when relevant"""

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.llm.invoke(prompt))
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "success": True,
                "question": question,
                "answer": answer,
                "document_id": document_id,
                "filename": doc_context['filename']
            }
        except Exception as e:
            print(f"PDF Q&A failed: {e}")
            return {
                "success": False,
                "error": f"Failed to answer question: {str(e)}"
            }
    
    async def _generate_ai_analysis(self, text: str, doc_type: str, metrics: Dict) -> str:
        """Generate AI analysis of the document."""
        if not self.llm:
            return self._generate_fallback_analysis(doc_type, metrics)
        
        doc_type_labels = {
            "earnings_report": "Quarterly/Annual Earnings Report",
            "annual_report": "Annual Report",
            "balance_sheet": "Balance Sheet",
            "income_statement": "Income Statement / P&L",
            "cash_flow": "Cash Flow Statement",
            "portfolio_statement": "Portfolio/Investment Statement",
            "research_report": "Stock Research Report",
            "mutual_fund_statement": "Mutual Fund Statement",
            "demat_statement": "Demat Account Statement",
            "general_financial": "Financial Document"
        }
        
        prompt = f"""You are a financial analyst. Analyze this {doc_type_labels.get(doc_type, 'financial document')} and provide clear insights.

Document Content:
{text}

Extracted Metrics: {metrics if metrics else 'None found'}

Please provide:
1. **Document Summary** - What is this document about?
2. **Key Findings** - Important numbers and facts
3. **Financial Health Assessment** - Is it positive, negative, or neutral?
4. **Investment Implications** - What does this mean for investors?
5. **Recommendations** - Any actionable suggestions

Format your response with clear headers and bullet points. Be concise but thorough."""

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.llm.invoke(prompt))
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"AI analysis failed: {e}")
            return self._generate_fallback_analysis(doc_type, metrics)
    
    def _generate_fallback_analysis(self, doc_type: str, metrics: Dict) -> str:
        """Generate basic analysis without AI."""
        analysis = f"## Document Analysis\n\n"
        analysis += f"**Document Type:** {doc_type.replace('_', ' ').title()}\n\n"
        
        if metrics:
            analysis += "### Extracted Financial Metrics\n\n"
            for key, value in metrics.items():
                formatted_key = key.replace("_", " ").title()
                if key in ["revenue", "profit", "market_cap"]:
                    analysis += f"- **{formatted_key}:** ₹{value:,.2f}\n"
                elif key in ["growth"]:
                    analysis += f"- **{formatted_key}:** {value}%\n"
                else:
                    analysis += f"- **{formatted_key}:** {value}\n"
            analysis += "\n"
        else:
            analysis += "No specific financial metrics were automatically extracted.\n\n"
        
        analysis += "### Note\n"
        analysis += "For detailed AI-powered analysis, please ensure the AI service is properly configured.\n"
        
        return analysis


# Create singleton instance
pdf_service = PDFService()
