from pathlib import Path
from langchain_core.documents.base import Document
from PIL import Image
from re import findall
from pymupdf4llm import to_markdown
from unicodedata import normalize
import os
from numpy import array

class Markdown_Image:

    old_text:str
    text:str
    images:list[Image.Image]

    def __init__(
            self,
            old_text:str,
            images:list[Image.Image]
            ) -> None:
        
        self.images = images
        self.old_text = old_text
        self.text = old_text

        self.image_merge_treshold = 0.3

    def __save_image_and_add_filename(self,image:Image.Image,image_dir:Path) -> Image.Image:
        
        parent_dir = str(image_dir.parent)

        new_path = f'{parent_dir}/new0.png'
        i = 1
        while os.path.exists(new_path):
            new_path = f'{parent_dir}/new{i}.png'
            i += 1

        image.filename = new_path
        image.save(new_path)

        return image


    def __merge_two_images(
            self,
            img1:Image.Image, 
            img2:Image.Image
            ) -> Image.Image:
        width = max((img1.width,img2.width))

        new = Image.new(
            "RGB",
            (width,img1.height+img2.height),
            "white"
            )
        
        new.paste(img1,(0,0))
        new.paste(img2,(0,img1.height))

        new = self.__save_image_and_add_filename(new,Path(img2.filename))

        os.remove(img1.filename)
        os.remove(img2.filename)
        
        return new


    def __are_image_connectable(
            self,
            img1:Image.Image, 
            img2:Image.Image
            ) -> bool:
        width = min((img1.width,img2.width))

        hit_count = 0

        img1 = array(img1)
        img2 = array(img2)

        for i in range(width):

            if  img1[-1][i][0] == img2[0][i][0] and\
                img1[-1][i][1] == img2[0][i][1] and\
                img1[-1][i][2] == img2[0][i][2]:

                hit_count += 1

        return (hit_count/width)>self.image_merge_treshold


    def connect_all_images_if_possible(self) -> None:

        i = 1
        
        while i < len(self.images):

            if self.__are_image_connectable(self.images[i-1],self.images[i]):
                
                self.images[i] = self.__merge_two_images(self.images[i-1],self.images[i])
                
                self.images.remove(self.images[i-1])
                self.text = '\n'.join([f'![]({image.filename})' for image in self.images])

            else:

                i += 1

    def __str__(self):
        
        return self.text


class Data_Loader:


    def __init__(
            self,
            images_dir:Path=Path('./images/')
            ) -> None:
        
        self.images_dir = images_dir


    def __find_all_images_in_markdown(self,markdown:str) -> list[Markdown_Image]:

        # Matches one or more consecutive ![](images/....png) blocks
        block_pattern = (
            r"(?:!\[\]\(images\/[^\/]+\.pdf-\d{4}-\d{2}\.png\))"
            r"(?:\s+!\[\]\(images\/[^\/]+\.pdf-\d{4}-\d{2}\.png\))*"
        )

        # Extracts just the path from a single ![](path) token
        path_pattern = r"!\[\]\(([^)]+)\)"

        result = []

        for block in findall(block_pattern,markdown):
            images_list = findall(path_pattern,block)

            result.append(
                Markdown_Image(
                    block,
                    [Image.open(image_path) for image_path in images_list]
                )
            )

        return result
    

    def __get_all_files_from_folder(self,path:Path) -> list[Path]:

        result = []

        if not os.path.exists(path): return []

        for directory in os.walk(path):

            files_full_paths:list[Path] = [Path(f'./{directory[0]}/{file_name}') for file_name in directory[-1]]
            
            result = result + files_full_paths
        
        return result
    

    def __get_file_extension(self,file_path:Path) -> str:

        return str(file_path).split('.')[-1]

    
    def __md_handler(self,path:Path) -> Document:
        
        with open(path,'r',encoding='UTF-8') as file:
            content = file.read()

        return Document(
            page_content=content,
            metadata={'source':str(path)}
        )

    def __pdf_handler(self,path:Path) -> Document:
        
        file_content = to_markdown(
            str(path),
            write_images=True,
            image_path=str(self.images_dir),
            image_format='png',
            dpi=300
        )

        file_content = normalize("NFC", file_content)

        for image in self.__find_all_images_in_markdown(file_content):

            image.connect_all_images_if_possible()

            file_content = file_content.replace(
                image.old_text,
                image.text
            )

        return Document(
            page_content=file_content,
            metadata={'source':str(path)}
        )

    FILE_HANDLERS = {
        "md":__md_handler,
        "pdf":__pdf_handler
    }

    def load(self,path:Path) -> list[Document]:

        result = []

        files = self.__get_all_files_from_folder(path)

        for file in files:

            result.append(
                self.FILE_HANDLERS[self.__get_file_extension(file)](self,path=file)
            )

        return result

