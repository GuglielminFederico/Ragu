import yaml
import logging
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    TableStructureOptions,
    PdfPipelineOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ConfigurablePipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def build_pipeline_options(self):
        cfg = self.config
        options = PdfPipelineOptions()
        
        options.create_legacy_output = cfg.get("create_legacy_output", True)
        options.document_timeout = cfg.get("document_timeout", True)
        options.enable_remote_services = cfg.get("enable_remote_services", False)
        options.allow_external_plugins = cfg.get("allow_external_plugins", False)
        options.artifacts_path = cfg.get("artifacts_path", None)
        options.do_ocr = cfg.get("ocr", False)
        options.do_table_structure = cfg.get("table_structure", False)
        options.generate_page_images = cfg.get("generate_page_images", False)
        options.generate_picture_images = cfg.get("generate_picture_images", False)
        options.images_scale = cfg.get("images_scale", 1.0)
        options.do_code_enrichment = cfg.get("code_enrichment", False)
        options.do_formula_enrichment = cfg.get("formula_enrichment", False)
        options.do_picture_classification = cfg.get("picture_classification", False)
        options.do_picture_description = cfg.get("picture_description", False)
        options.force_backend_text = cfg.get("force_backend_text", False)
        options.generate_table_images = cfg.get("generate_table_images", False)
        options.generate_parsed_images = cfg.get("generate_parsed_images", False)

        table_cfg = cfg.get("TableStructureOptions", {})
        options.table_structure_options = TableStructureOptions(
            do_cell_matching=table_cfg.get("do_cell_matching", False),
            mode=table_cfg.get("mode", "accurate"),
        )

        ocr_cfg = cfg.get("ocr_options", {})
        options.ocr_options = {
            "lang": ocr_cfg.get("lang", ["fr", "de", "es", "en"]),
            "force_full_page_ocr": ocr_cfg.get("force_full_page_ocr", False),
            "bitmap_are_threshold": ocr_cfg.get("bitmap_are_threshold", 0.05),
            "use_gpu": ocr_cfg.get("use_gpu", None),
            "confidence_threshold": ocr_cfg.get("confidence_threshold", 0.5),
            "model_storage_directory": ocr_cfg.get("model_storage_directory", None),
            "recog_network": ocr_cfg.get("recog_network", "standard"),
            "download_enabled": ocr_cfg.get("download_enabled", True)
        }

        pic_desc_cfg = cfg.get("picture_description_options", {})
        options.picture_description_options = {
            "batch_size": pic_desc_cfg.get("batch_size", 8),
            "scale": pic_desc_cfg.get("scale", 2),
            "picture_area_threshold": pic_desc_cfg.get("picture_area_threshold", 0.05),
            "repo_id": pic_desc_cfg.get("repo_id", "HuggingFaceTB/SmolVLM-256M-Instruct"),
            "prompt": pic_desc_cfg.get("prompt", "Describe this image in a few sentences."),
            "generation_config": pic_desc_cfg.get("generation_config", {
                "max_new_tokens": 200,
                "do_sample": True
            })
        }

        accel_cfg = cfg.get("AcceleratorOptions", {})
        device = accel_cfg.get("device", "auto").lower()
        device_enum = {
            "cpu": AcceleratorDevice.CPU,
            "gpu": AcceleratorDevice.GPU,
            "auto": AcceleratorDevice.AUTO
        }.get(device, AcceleratorDevice.AUTO)

        options.accelerator_options = AcceleratorOptions(
            num_threads=accel_cfg.get("num_threads", -1),
            device=device_enum,
            cuda_use_flash_attention2 = accel_cfg.get("cuda_use_flash_attention2", False),
        )

        return options

    def get_document_converter(self):
        pipeline_options = self.build_pipeline_options()
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

