import os
import hashlib
import yaml
import json
import logging
from pathlib import Path
from os import listdir
from os.path import isfile, join
from Converter import ConfigurablePipeline
from docling_core.types.doc import PictureItem
from docling.datamodel.document import ConversionStatus

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class KnowledgeBase:

    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.input_path = self.config.get("input_path", None)
        self.output_path = self.config.get("output_path", None)
        self.image_path = self.config.get("image_path", None)
        self.file_list = self.config.get("file_list", [])
        self.docling_config = self.config.get("docling_config", None)
        self.text_keys = {'self_ref', 'parent', 'label', 'prov', 'text_content'}
        self.table_keys = {'self_ref', 'parent', 'label', 'prov', 'captions', 'num_rows', 'num_cols', 'data'}
        self.image_keys = {'self_ref', 'parent', 'label', 'prov', 'captions'}
        self.cell_keys = {'start_row_offset_idx',  'start_col_offset_idx', 'text', 'column_header'}

    @staticmethod
    def image_to_text(image_path):
        """Describe image using VL models."""

        desc_placeholder = "Placeholder for image-to-text"

        image_desc = {}
        for filename in os.listdir(image_path):
            file_path = os.path.join(image_path, filename)
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_ref = filename.split('/')[-1]
                image_desc[image_ref] = desc_placeholder

        with open('image_descr.json', 'w') as f:
            json.dump(image_desc, f)
            
    @staticmethod
    def filter_keys(elements, keys_to_keep):
        return [{k: v for k, v in elem.items() if k in keys_to_keep} for elem in elements]

    # Rename 'text' key, PyMilvus is not accepting it
    @staticmethod
    def replace_text_key(docling_dic):
        docling_dic['text_content'] = docling_dic.pop('text')
        return docling_dic

    #make sure that tabular data is properly ordered
    @staticmethod
    def sort_table_elements(elements):
        return sorted(elements, key=lambda x: (x['start_row_offset_idx'], x['start_col_offset_idx']))

    @staticmethod
    def sort_elements(elements):
        return sorted(elements, key=lambda x: (x['prov'][0]['page_no'], -x['prov'][0]['bbox']['t']))

    # Creating nested elements
    @staticmethod
    def nesting_table(input_dict):

        transformed_table = {
            "self_ref": input_dict["self_ref"],
            "parent": input_dict["parent"],
            "label": input_dict["label"],
            "document": input_dict["document"],
            "page": input_dict["page"],
            "text_content": input_dict["text_content"],
            "table_metadata": {
                "header": input_dict["header"],
                "content": input_dict["content"],
                "caption": input_dict.get("caption", ""),
                "num_columns": input_dict.get("num_columns", ""),
                "num_rows": input_dict.get("num_rows", "")
            }
        }
        return transformed_table

    @staticmethod
    def nesting_image(input_dict):

        transformed_image = {
            "self_ref": input_dict["self_ref"],
            "parent": input_dict["parent"],
            "label": input_dict["label"],
            "document": input_dict["document"],
            "page": input_dict["page"],
            "text_content": input_dict["text_content"],
            "image_metadata": {
                "image_caption": input_dict["image_caption"]
            }
        }
        return transformed_image
    
    @staticmethod
    def propagate_titles(documents):
        current_title = [doc['text_content'] for doc in documents if doc['label']=='section_header'][0]
        for entry in documents:
            if entry.get("label") == "section_header":
                current_title = entry["text_content"]
            entry["title"] = current_title
        return documents
    
    @staticmethod
    def generate_unique_id(element):
        """Generates a unique hash based on self_ref and document fields."""
        key_fields = {
            "self_ref": element.get("self_ref", ""),
            "document": element.get("document", "")
        }
        key_str = json.dumps(key_fields, sort_keys=True)
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()
    
    def process_documents(self, results):

        """Process documents and save results."""
        output_path = Path(self.output_path)
        image_path = Path(self.image_path)

        for res in results:
            if res.status == ConversionStatus.SUCCESS:

                doc_filename = res.input.file.stem

                picture_counter = 0
                for element, _ in res.document.iterate_items():
                    if isinstance(element, PictureItem):
                        picture_counter += 1
                        element_image_filename = (
                            image_path / f"{doc_filename}-picture-{picture_counter}.png"
                        )
                        with element_image_filename.open("wb") as fp:
                            element.get_image(res.document).save(fp, "PNG")
                _log.info(f"\t{picture_counter} pictures extracted")

                #Export Json
                with (output_path / f"{doc_filename}.json").open("w") as fp:
                    fp.write(json.dumps(res.document.export_to_dict()))

                #Export Markdown
                with (output_path / f"{doc_filename}.md").open("w") as fp:
                    fp.write(res.document.export_to_markdown())   

            elif res.status == ConversionStatus.PARTIAL_SUCCESS:
                _log.info(
                    f"Document {res.input.file} was partially converted with the following errors:"
                )  
                for item in res.errors:
                    _log.info(f"\t{item.error_message}")
            
            else:
                _log.info(f"Document {res.input.file} failed to convert.")

    def create_knowledge(self):
        """Create document knowledge base."""

        output_path = Path(self.output_path)
        image_path = Path(self.image_path)
        output_path.mkdir(parents=True, exist_ok=True)
        image_path.mkdir(parents=True, exist_ok=True)

        source_files = [f for f in listdir(self.input_path) if isfile(join(self.input_path, f))]
        if self.file_list:
            source_files = [f for f in source_files if f in self.file_list]

        doc_paths = [Path(join(self.input_path, f)) for f in source_files]

        docling_pipeline = ConfigurablePipeline(config_path=self.docling_config)
        doc_converter = docling_pipeline.get_document_converter()

        results = doc_converter.convert_all(
            doc_paths,
            raises_on_error=False
        )

        self.process_documents(results)
        self.image_to_text(image_path)

    def refine_knowledge(self):
        """Refine document knowledge base."""
        image_descr = json.load(open('image_descr.json'))

        for filename in os.listdir(self.output_path):
            file_path = os.path.join(self.output_path, filename)
            if os.path.isfile(file_path) and file_path.lower().endswith(('.json')):
                
                doc_name = file_path.split('/')[-1].replace('.json', '')
                print(f'Processing {doc_name}')
                doc_json = json.load(open(file_path))
                texts = doc_json['texts'].copy()
                #see if we need to remove page_header and page_footer TODO
                #texts = [text for text in texts if text['label'] not in ['page_footer', 'page_header']]
                tables = doc_json['tables'].copy()
                images = doc_json['pictures'].copy()

                texts = [self.replace_text_key(text) for text in texts]
            
                # Keys to keep
                #keep also bounding box information to aggregate text and link to image/table, it's inside prov
                #inside prov, we have also char_span, keep it to measure the text length

                # Filtered list
                filtered_texts = self.filter_keys(texts, self.text_keys)
                filtered_tables = self.filter_keys(tables, self.table_keys)
                filtered_images = self.filter_keys(images, self.image_keys)
                
                for table in filtered_tables:
                    table['table_data'] = (self.filter_keys(table['data']['table_cells'], self.cell_keys))
                    table.pop('data', None)

                for table in filtered_tables:
                    table['table_data'] = self.sort_table_elements(table['table_data'])

                # retrieve table columns and table content
                # maybe save cell bboxes as well TODO
                for table in filtered_tables:

                    table['header'] = []
                    table['content'] = []
                    if table['table_data']: 
                        table['num_columns'] = max([x['start_col_offset_idx'] for x in table['table_data']])
                        table['num_rows'] = max([x['start_row_offset_idx'] if table['table_data'] else 0 for x in table['table_data']])
                    else:
                        table['num_columns'] = 0
                        table['num_rows'] = 0
                        
                    table['content'] = [[] for _ in range(table['num_rows'] + 1)]

                    for cell in table['table_data']:
                        
                        if cell['column_header']:
                            table['header'].append(cell['text']) 

                        table['content'][cell['start_row_offset_idx']].append(cell['text'])

                    table['content'] = [' | '.join(row) for row in table['content']]    
                    table.pop('table_data', None)

                filtered_elements = filtered_texts + filtered_tables + filtered_images

                # add document name to each element
                for element in filtered_elements:
                    element['document'] = doc_json['name']

                sorted_elements = self.sort_elements(filtered_elements)

                # add page number to each element
                for element in sorted_elements:
                    element['page'] = element['prov'][0]['page_no']
                    element.pop('prov', None)

                # add caption to each element
                captions = {}
                for element in sorted_elements:
                    if element['label']=='caption':
                        captions[element['self_ref']] = element['text_content'] 

                #replace captions with corresponding text
                for element in sorted_elements:
                    if 'captions' in element.keys():
                        if element['captions']:
                            element['captions'] = captions[element['captions'][0]['$ref']]
                            
                for element in sorted_elements:
                    if 'pictures' in element['self_ref']:
                        element['image_caption'] = element.pop('captions')
                    if 'tables' in element['self_ref']:
                        element['table_caption'] = element.pop('captions') 
                        
                for element in sorted_elements:
                    element['parent'] = element['parent']['$ref']

                for element in sorted_elements:
                    if 'pictures' in element['self_ref']:
                        image_nr = element['self_ref'].split('/')[-1]
                        image_ref = f'{doc_name}-picture-{image_nr}.png'
                        if image_ref in image_descr.keys():
                            element['text_content'] = image_descr[image_ref]
                        else:
                            element['text_content'] = 'placeholder for image-to-text'
                    if 'tables' in element['self_ref']:
                        element['text_content'] = '' 
                        
                for i, element in enumerate(sorted_elements):
                    if 'pictures' in element['self_ref']:
                        sorted_elements[i] = self.nesting_image(element)
                    if 'tables' in element['self_ref']:
                        sorted_elements[i] = self.nesting_table(element)

                sorted_elements = self.propagate_titles(sorted_elements)
                #create unique id for each element
                for el in sorted_elements:
                    el["_id"] = self.generate_unique_id(el)
                    
                with open('refined_kb.json', 'w') as f:
                    json.dump(sorted_elements, f, indent=2)
                #column renaming according to AIDA
                # for elem in sorted_elements:
                    
                #     elem['document_name'] = elem.pop('document')
                #     elem['extracted_text'] = elem.pop('text_content')
                #     elem['text_label'] = elem.pop('label')
                #     elem['associated_title_or_section'] = elem.pop('title')
                #     if elem['text_label'] == 'picture':
                #         elem['figure_metadata'] = elem.pop('image_metadata')
                #     elem['extracted_text_type'] = 'original' if elem['text_label'] not in ['picture', 'table'] else 'generated'
                #     elem['element_id'] = ' ' if elem['text_label'] not in ['picture', 'table'] else elem['self_ref']
                #     elem['merge_with_next_text'] = False
                #     elem.pop('self_ref')
                #     elem.pop('parent')


    def delete_knowledge(self, key):
        """Remove document from knowledgebase."""
        return self.knowledge.get(key, "Knowledge not found.")
    
    def upload_knowledge(self, key):
        """Upload document to knowledgebase."""
        return self.knowledge.get(key, "Knowledge not found.")
