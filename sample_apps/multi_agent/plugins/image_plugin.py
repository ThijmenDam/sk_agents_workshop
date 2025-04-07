import os
from typing import Annotated
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents import ImageContent
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures

class ImageAnalysisPlugin:
    """A plugin that analyzes images using Azure Computer Vision."""
    def __init__(self):
        self.client = ImageAnalysisClient(
            endpoint=os.environ["COMPUTER_VISION_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["COMPUTER_VISION_KEY"]),
        )

    @kernel_function(description="Analyze an image and generate a detailed caption")
    async def extract_image_from_message(self, message: Annotated[ImageContent, "The image_data inside the inner_content"]) \
        -> Annotated[str, "Image description"]:
        
        print(f"Message content: {type(message.data)} {message.data}")

        try:
            # Analyze the image with multiple visual features
            result = self.client.analyze(
                image_data=message.data,
                visual_features=[
                    VisualFeatures.CAPTION,
                    VisualFeatures.TAGS,
                    VisualFeatures.OBJECTS,
                    VisualFeatures.DENSE_CAPTIONS
                ],
                gender_neutral_caption=True,
            )

            # Build a comprehensive description
            description_parts = []
            
            if result.caption:
                description_parts.append(result.caption.text)
            
            if result.tags:
                relevant_tags = [tag.name for tag in result.tags[:5]]
                description_parts.append(f"Key features: {', '.join(relevant_tags)}")
            
            if result.dense_captions:
                detailed_captions = [cap.text for cap in result.dense_captions[:3]]
                description_parts.append("Additional details: " + "; ".join(detailed_captions))

            return " ".join(description_parts) if description_parts else "No description could be generated for this image."

        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return f"Error analyzing image: {str(e)}"