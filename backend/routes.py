

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .models import get_model, get_gradcam
from .gradcam import create_heatmap_overlay
from .utils import preprocess_image, numpy_to_base64, base64_to_data_uri, check_image_quality
from .schemas import PredictionResponse, HealthResponse
from .lung_segmentation import segment_lungs, validate_lung_mask


router = APIRouter()

# Decision threshold - LOWERED for high recall (fewer false negatives)
# Default is 0.5, we use 0.3 to catch more pneumonia cases
DECISION_THRESHOLD = 0.3


def get_risk_level(confidence: float) -> str:
    """
    Classify risk based on pneumonia probability.
    
    Low threshold (0.3) means we're conservative - 
    anything above 0.3 is flagged for doctor review.
    """
    if confidence < 0.3:
        return "Low Risk"
    elif confidence < 0.6:
        return "Medium Risk - Review Recommended"
    else:
        return "High Risk - Urgent Review"


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Analyze chest X-ray for pneumonia.
    
    Accepts: JPEG/PNG image file
    Returns: Prediction, confidence, and Grad-CAM heatmap
    
    High recall mode: Uses 0.3 threshold to minimize false negatives.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG)")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Preprocess image
        tensor, original_image = preprocess_image(image_bytes)
        
        # Run image quality check (lightweight, before inference)
        quality = check_image_quality(original_image)
        
        # Get model (loaded once, cached)
        model, device = get_model()
        tensor = tensor.to(device)
        
        # Get Grad-CAM singleton (hooks registered once at startup)
        gradcam = get_gradcam()
        
        # Generate prediction and heatmap
        cam, raw_confidence = gradcam.generate(tensor)
        
        # Apply decision threshold for classification
        # raw_confidence = P(pneumonia)
        is_pneumonia = raw_confidence >= DECISION_THRESHOLD
        prediction = "Pneumonia" if is_pneumonia else "Normal"
        
        # Run lung segmentation to constrain heatmap to lung regions
        lung_mask = segment_lungs(original_image)
        mask_validation = validate_lung_mask(lung_mask)
        
        # Create heatmap overlay WITH lung mask
        # Colors only appear where CAM is significant AND inside lungs
        overlay = create_heatmap_overlay(original_image, cam, lung_mask=lung_mask, alpha=0.6)
        
        # Convert to base64 for API response
        heatmap_base64 = numpy_to_base64(overlay)
        heatmap_data_uri = base64_to_data_uri(heatmap_base64)
        
        # Generate doctor-friendly note
        if is_pneumonia:
            note = (
                f"AI detected potential pneumonia with {raw_confidence:.1%} confidence. "
                f"Highlighted regions show model attention within lung boundaries. "
                f"Please review for clinical confirmation."
            )
        else:
            note = (
                f"No significant pneumonia indicators detected (confidence: {raw_confidence:.1%}). "
                f"This is an AI screening - clinical judgment should prevail."
            )
        
        # Add lung segmentation warnings if any
        for warning in mask_validation.get("warnings", []):
            note += f" Segmentation warning: {warning}"
        
        # Append quality warning if applicable (informational only)
        if quality["warning"]:
            note = note + " " + quality["warning"]
        
        return PredictionResponse(
            prediction=prediction,
            confidence=round(raw_confidence, 4),
            risk_level=get_risk_level(raw_confidence),
            heatmap_image=heatmap_base64,
            heatmap_data_uri=heatmap_data_uri,
            device=str(device),
            low_image_quality=quality["is_low_quality"],
            note=note
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns model status and device info.
    """
    try:
        model, device = get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=model is not None,
            device=str(device)
        )
    except Exception as e:
        return HealthResponse(
            status=f"unhealthy: {str(e)}",
            model_loaded=False,
            device="unknown"
        )
