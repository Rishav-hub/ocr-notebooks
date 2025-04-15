import pandas as pd
import xml.etree.ElementTree as ET
def OCR_XML_to_DataFrame(XML_file_path, req_fields):
    df = pd.DataFrame()
    row = 0
    single_fields = []
    grp_names = []
    grp_field_names = []
    tab_names = []
    table_field_names = []
    field_to_extract = 'None'

    if ".TRANS" or ".KEY" in XML_file_path:
        tree = ET.parse(XML_file_path)
        root = tree.getroot()
        groupNames = root.findall("./FldColls/FldSet")
        w_h = root.find("./Imgs/Img")
        image_name = w_h.attrib["Src"]
        width = w_h.attrib["w"]
        height = w_h.attrib["h"]
        
        images = []
        for img in root.findall("./Imgs/Img"):
            image_name = img.attrib["Src"]
            page_num = int(img.attrib["Pg"])
            images.append((image_name, page_num))
        
        for gn in groupNames:
            fields = gn.findall("./Fld")
            for field in fields:
                if field.attrib["Typ"] == "TABLE":
                    tabs_name = field.find("./Nm").text
                    tab_names.append(tabs_name)
#         print(tab_names)
        for gn in groupNames:
            fields = gn.findall("./Fld")
            for field in fields:
                if field.attrib["Typ"] == "SINGLE":
                    Sin_field_name = field.find("./Nm").text
                    single_fields.append(Sin_field_name)

                    # If required to Extract Specific Single Line Fields, pass the single line fileds list here
                    if Sin_field_name in req_fields: 
                        work_s = field.findall("./Ln/Work")
                        history_s = field.findall("./Ln/Hstry")
                        valset_s = field.findall("./Ln/ValSet")
                           
                        if len(work_s) == 0:
                            work_s = field.findall("./Ln/Clm/Work")
                            history_s = field.findall("./Ln/Clm/Hstry")
                            valset_s = field.findall("./Ln/Clm/ValSet")
                            
                        for s in range(len(work_s)):
                            data_s = work_s[s].findall("./Data")
                            sin_Trans_data = work_s[s].findall("./Trans")

                        try:
                            Template_type = sin_Trans_data[0].text
                        except:
                            Template_type = data_s[0].text
                        
                        for each_single_row in data_s:
                            df.loc[row, "Data"] = each_single_row.text

                        for each_sin_row in sin_Trans_data:
                            df.loc[row, "TRANS_Value"] = each_sin_row.text
                        
                        for c in range(len(valset_s)):
                            comments = valset_s[c].findall("./Val")
                            
                        for each_com in comments:
                            df.loc[row, "Comments"] = each_com.text
                        
                        for k in range(len(history_s)):
                            cords = work_s[k].find("./Crds")
                            data_s = history_s[k].findall("./Data")
                            df.loc[row, "Group_name"] = Sin_field_name
                            df.loc[row, "Image_width"] = w_h.attrib["w"]
                            df.loc[row, "Image_height"] = w_h.attrib["h"]
                            df.loc[row, "Field_Name"] = Sin_field_name
                            df.loc[row, "x1"] = cords.attrib["x1"]
                            df.loc[row, "x2"] = cords.attrib["x2"]
                            df.loc[row, "y1"] = cords.attrib["y1"]
                            df.loc[row, "y2"] = cords.attrib["y2"]
                            for sin_row_h in data_s:
                                    if sin_row_h.attrib["Src"] == "OCR_OM":
                                        df.loc[row, "OCR_OMNI"] = sin_row_h.text
                                    
                                    if sin_row_h.attrib["Src"] == "OCR_IG":
                                        df.loc[row, "OCR_IGEAR"] = sin_row_h.text
                                        
                                    if sin_row_h.attrib["Src"] == "OCR_AT":
                                        df.loc[row, "OCR_AT"] = sin_row_h.text
                                      
                                    if sin_row_h.attrib["Src"] == "OCROptimizer":
                                        df.loc[row, "OCR_Optimizer"] = sin_row_h.text

                        row += 1

                elif field.attrib["Typ"] == "GROUP":
                    grp_name = field.find("./Nm").text
                    grp_names.append(grp_name)
    #                 print(":::Group Name:::", grp_name)
                    grp_fields = field.findall("./Ln/Clm")
                    for idx, grp_field in enumerate(grp_fields):
    #                     print(idx, grp_field.attrib["Nm"])
                        grp_field_names.append(grp_field.attrib["Nm"])
                    
            # If required to Extract Specific Group Line Fields, pass the single line fileds list here
                        if grp_field.attrib["Nm"] in req_fields:
                            work_g = grp_field.findall("./Ln/Work")
                            history_g = grp_field.findall("./Ln/Hstry")
                            valset_g = grp_field.findall("./Ln/ValSet")
                            
                            if len(work_g) == 0:
                                work_g = grp_field.findall("./Work")
                                history_g = grp_field.findall("./Hstry")
                                valset_g = grp_field.findall("./ValSet")

                            for k in range(len(work_g)):
                                group_data = work_g[k].findall("./Data")
                                Trans_data = work_g[k].findall("./Trans")

                            for each_grp_row in group_data:
                                df.loc[row, "Data"] = each_grp_row.text

                            for each_trans_row in Trans_data:
                                df.loc[row, "TRANS_Value"] = each_trans_row.text

                            for gc in range(len(valset_g)):
                                comments_g = valset_g[gc].findall("./Val")
                                
                            for each_com_g in comments_g:
                                df.loc[row, "Comments"] = each_com_g.text
                            
                            for H in range(len(history_g)):
                                data_gh = history_g[H].findall("./Data")
                                cords = work_g[k].find("./Crds")
                                df.loc[row, "Group_name"] = grp_name
                                df.loc[row, "Image_width"] = w_h.attrib["w"]
                                df.loc[row, "Image_height"] = w_h.attrib["h"]
                                df.loc[row, "Field_Name"] = grp_field.attrib["Nm"]
                                df.loc[row, "x1"] = cords.attrib["x1"]
                                df.loc[row, "x2"] = cords.attrib["x2"]
                                df.loc[row, "y1"] = cords.attrib["y1"]
                                df.loc[row, "y2"] = cords.attrib["y2"]
                                for data_in_grprow in data_gh:
                                    if data_in_grprow.attrib["Src"] == "OCR_OM":
                                        df.loc[row, "OCR_OMNI"] = data_in_grprow.text
                                            
                                    if data_in_grprow.attrib["Src"] == "OCR_IG":
                                        df.loc[row, "OCR_IGEAR"] = data_in_grprow.text
                                        
                                    if data_in_grprow.attrib["Src"] == "OCR_AT":
                                        df.loc[row, "OCR_AT"] = data_in_grprow.text    
                                     
                                    if data_in_grprow.attrib["Src"] == "OCROptimizer":
                                        df.loc[row, "OCR_Optimizer"] = data_in_grprow.text
                                
                            row += 1

                elif field.attrib["Typ"] == "TABLE":
                    tab_name = field.find("./Nm").text
                    if ("ClmServiceInfoTable_BAK" in tab_names) and (tab_name  == "ClmServiceInfoTable_BAK"):
                        table_fields = field.findall("./Ln/Clm")
                        for idx, tab_field in enumerate(table_fields):
                            table_field_names.append(tab_field.attrib["Nm"])

                            if tab_field.attrib["Nm"] in req_fields:
                                work_t = tab_field.findall("./Ln/Work")
                                history_t = tab_field.findall("./Ln/Hstry")
                                valset_t = tab_field.findall("./Ln/ValSet")

                                if len(work_t) == 0:
                                    work_t = tab_field.findall("./Work")
                                    history_t = tab_field.findall("./Hstry")
                                    valset_t = tab_field.findall("./ValSet")

                                for l in range(len(work_t)):
                                    tab_data = work_t[l].findall("./Data")
                                    tab_Trans_data = work_t[l].findall("./Trans")

                                for each_tab_row in tab_data:
                                    df.loc[row, "Data"] = each_tab_row.text

                                for each_tab_trans_row in tab_Trans_data:
                                    df.loc[row, "TRANS_Value"] = each_tab_trans_row.text

                                for tc in range(len(valset_t)):
                                    comments_t = valset_t[tc].findall("./Val")

                                for each_com_t in comments_t:
                                    df.loc[row, "Comments"] = each_com_t.text                            


                                for T in range(len(history_t)):
                                    data_th = history_t[T].findall("./Data")
                                    cords = work_t[T].find("./Crds")
                                    df.loc[row, "Group_name"] = tab_field.attrib["Nm"]
                                    df.loc[row, "Image_width"] = w_h.attrib["w"]
                                    df.loc[row, "Image_height"] = w_h.attrib["h"]
                                    df.loc[row, "Field_Name"] = tab_field.attrib["Nm"]
                                    df.loc[row, "x1"] = cords.attrib["x1"]
                                    df.loc[row, "x2"] = cords.attrib["x2"]
                                    df.loc[row, "y1"] = cords.attrib["y1"]
                                    df.loc[row, "y2"] = cords.attrib["y2"]
                                    for data_in_tabrow in data_th:
                                        if data_in_tabrow.attrib["Src"] == "OCR_OM":
                                            df.loc[row, "OCR_OMNI"] = data_in_tabrow.text

                                        if data_in_tabrow.attrib["Src"] == "OCR_IG":
                                            df.loc[row, "OCR_IGEAR"] = data_in_tabrow.text

                                        if data_in_tabrow.attrib["Src"] == "OCR_AT":
                                            df.loc[row, "OCR_AT"] = data_in_tabrow.text      

                                        if data_in_tabrow.attrib["Src"] == "OCROptimizer":
                                            df.loc[row, "OCR_Optimizer"] = data_in_tabrow.text
                                row += 1
                    elif ("ClmServiceInfoTable_BAK" in tab_names) and (tab_name == "ClmServiceInfoTable"):
                        pass
                    elif tab_name == "DEN_MissingToothTable":
                        pass
                    else:
                        table_fields = field.findall("./Ln/Clm")
                        for idx, tab_field in enumerate(table_fields):
                            table_field_names.append(tab_field.attrib["Nm"])
                            if tab_field.attrib["Nm"] in req_fields:
                                work_t = tab_field.findall("./Ln/Work")
                                history_t = tab_field.findall("./Ln/Hstry")
                                valset_t = tab_field.findall("./Ln/ValSet")

                                if len(work_t) == 0:
                                    work_t = tab_field.findall("./Work")
                                    history_t = tab_field.findall("./Hstry")
                                    valset_t = tab_field.findall("./ValSet")

                                for l in range(len(work_t)):
                                    tab_data = work_t[l].findall("./Data")
                                    tab_Trans_data = work_t[l].findall("./Trans")

                                for each_tab_row in tab_data:
                                    df.loc[row, "Data"] = each_tab_row.text

                                for each_tab_trans_row in tab_Trans_data:
                                    df.loc[row, "TRANS_Value"] = each_tab_trans_row.text

                                for tc in range(len(valset_t)):
                                    comments_t = valset_t[tc].findall("./Val")

                                for each_com_t in comments_t:
                                    df.loc[row, "Comments"] = each_com_t.text                            


                                for T in range(len(history_t)):
                                    data_th = history_t[T].findall("./Data")
                                    cords = work_t[T].find("./Crds")
                                    df.loc[row, "Group_name"] = tab_field.attrib["Nm"]
                                    df.loc[row, "Image_width"] = w_h.attrib["w"]
                                    df.loc[row, "Image_height"] = w_h.attrib["h"]
                                    df.loc[row, "Field_Name"] = tab_field.attrib["Nm"]
                                    df.loc[row, "x1"] = cords.attrib["x1"]
                                    df.loc[row, "x2"] = cords.attrib["x2"]
                                    df.loc[row, "y1"] = cords.attrib["y1"]
                                    df.loc[row, "y2"] = cords.attrib["y2"]
                                    for data_in_tabrow in data_th:
                                        if data_in_tabrow.attrib["Src"] == "OCR_OM":
                                            df.loc[row, "OCR_OMNI"] = data_in_tabrow.text

                                        if data_in_tabrow.attrib["Src"] == "OCR_IG":
                                            df.loc[row, "OCR_IGEAR"] = data_in_tabrow.text

                                        if data_in_tabrow.attrib["Src"] == "OCR_AT":
                                            df.loc[row, "OCR_AT"] = data_in_tabrow.text      

                                        if data_in_tabrow.attrib["Src"] == "OCROptimizer":
                                            df.loc[row, "OCR_Optimizer"] = data_in_tabrow.text
                                row += 1           
                                
    return df, single_fields, grp_field_names, table_field_names, images