import os
from typing import Annotated
from semantic_kernel.functions import kernel_function
from semantic_kernel.contents import ImageContent
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
import aiohttp

class ImageAnalysisPlugin:
    """A plugin that analyzes images using Azure Computer Vision."""
    def __init__(self):
        self.client = ImageAnalysisClient(
            endpoint=os.environ["COMPUTER_VISION_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["COMPUTER_VISION_KEY"]),
        )   
                    
    async def download_image_bytes(self, url: str) -> bytes:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    raise Exception(f"Failed to download image. Status code: {response.status}")

    async def extract_image_from_message(self, message: Annotated[ImageContent, "Latest message from the user with the image"]) \
            -> Annotated[str, "Image description"]:

        try:    
            uri = str(message.uri)
            image_bytes = await self.download_image_bytes(uri)

            # Analyze image
            result = self.client.analyze(
                image_data=image_bytes,
                visual_features=[
                    VisualFeatures.CAPTION,
                    VisualFeatures.TAGS,
                    VisualFeatures.OBJECTS,
                    VisualFeatures.DENSE_CAPTIONS
                ],
                gender_neutral_caption=True,
            )

            # Build response
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
