# 创建一个csv，属性有pt, level, target, mccvsplit, slide_name，其中，straget从clinical.tsv中获取
import os
import pandas as pd
import argparse
import numpy as np
import random
import torch

# pt_pth = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/COAD_20'
# save_pth = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/COAD_20/csv'
# tsv_pth = '/mnt/SSD_4T/admin103/ydw/TCGA_download_txt/COAD/clinical.tsv'

# pt_pth = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/COAD_5'
# save_pth = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/COAD_5/csv'
# tsv_pth = '/mnt/SSD_4T/admin103/ydw/TCGA_download_txt/COAD/clinical.tsv'

# pt_pth = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/COAD_10'
# save_pth = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/COAD_10/csv'
# tsv_pth = '/mnt/SSD_4T/admin103/ydw/TCGA_download_txt/COAD/clinical.tsv'

# pt_pth = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/Camelyon16/Camelyon16_20'
# save_pth = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/Camelyon16/Camelyon16_20/csv'
# tsv_pth = '/mnt/SSD_4T/admin103/ydw/TCGA_download_txt/Camelyon16/clinical.tsv'

# pt_pth = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/IDC_P'
# save_pth = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/IDC_P/csv'
# tsv_pth = '/mnt/SSD_4T/admin103/ydw/TCGA_download_txt/IDC_P/clinical.tsv'

# pt_pth = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/IDC_P/anyi_Bx'
# save_pth = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/IDC_P/anyi_Bx/csv'
# tsv_pth = '/mnt/SSD_4T/admin103/ydw/TCGA_download_txt/anyi_Bx/clinical.tsv'

pt_pth = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/IDC_P/Beijing'
save_pth = '/mnt/SSD_4T/admin103/ydw/Slide_mil_classify/output/IDC_P/Beijing/csv'
tsv_pth = '/mnt/SSD_4T/admin103/ydw/TCGA_download_txt/IDC_P/beijing/clinical.tsv'



def make_csv_classify(pt_pth, targe_tsv,save_pth):
    files = os.listdir(pt_pth)
    # filer out non-file
    files = [file for file in files if os.path.isdir(os.path.join(pt_pth, file))]
    # open tsv file
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    tsv = pd.read_csv(targe_tsv, sep=',')

    for file in files:
        pth = os.path.join(pt_pth, file, 'pt_files')
        if not os.path.exists(pth):
            continue
        pt_files = os.listdir(pth)
        data = []

        saved_list = []
        for pt_file in pt_files:
            pt_file_path = os.path.join(pth, pt_file)
            pt_file_path_ = pt_file.split('.pt')[0]
            # Extract the slide name from the file name
            slide_name = pt_file_path_
            # Read the target from the tsv file
            target = tsv[tsv['filename'] == slide_name]['label']
            if not target.empty:
                target = target.values[0]
                if target == 'normal':
                    target = '0'
                elif target == 'tumor':
                    target = '1'
                else:
                    raise ValueError(f"Unknown target: {target}")
            else:
                target = 'unknown'
            
            # Append the data
            # 检查pt_file_path_是否存在于saved_list中,如果不存在，则加入data
            if pt_file_path_ in saved_list:
                continue
            else:
                saved_list.append(pt_file_path_)
                data.append({
                    'pt': pt_file_path_,
                    'slide_name': slide_name,
                    'target': target
                })

        # Create a DataFrame and save it as a CSV file
        df = pd.DataFrame(data)
        csv_file_path = os.path.join(save_pth, file + '.csv')
        # Split the DataFrame into training and validation sets
        # df = split_train_val(df)
        # Save the split DataFrame
        df.to_csv(csv_file_path, index=False)
        print(f"CSV file created: {csv_file_path}")

def make_csv_classify_COAD(pt_pth, targe_tsv,save_pth):
    files = os.listdir(pt_pth)
    # filer out non-file
    files = [file for file in files if os.path.isdir(os.path.join(pt_pth, file))]
    # open tsv file
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    tsv = pd.read_csv(targe_tsv, sep='\t')

    for file in files:
        pth = os.path.join(pt_pth, file, 'pt_files')
        if not os.path.exists(pth):
            continue
        pt_files = os.listdir(pth)
        data = []

        saved_list = []
        for pt_file in pt_files:
            pt_file_path = os.path.join(pth, pt_file)
            pt_file_path_ = pt_file.split('.pt')[0].split('_')[0]
            # Extract the slide name from the file name
            slide_name = pt_file.split('.')[-3].split('-')[0] + '-' + pt_file.split('.')[-3].split('-')[1] + '-' + pt_file.split('.')[-3].split('-')[2]
            # Read the target from the tsv file
            target = tsv[tsv['cases.submitter_id'] == slide_name]['diagnoses.ajcc_pathologic_t']
            if not target.empty:
                target = target.values[0]
                if target not in ["'--", '--']:
                    if target in ['T1', 'T1a', 'Tis'] :
                        target = '0'
                    elif target in ['T2', 'T2c']:
                        target = '1'
                    elif target in ['T3', 'T3a']:
                        target = '2'
                    elif target in ['T4', 'T4a', 'T4b']:
                        target = '3'
                        
                    # Append the data
                    # 检查pt_file_path_是否存在于saved_list中,如果不存在，则加入data
                    if pt_file_path_ in saved_list:
                        continue
                    else:
                        saved_list.append(pt_file_path_)
                        data.append({
                            'pt': pt_file_path_,
                            'slide_name': slide_name,
                            'target': target
                        })
            else:
                raise ValueError(f"Unknown target: {target}")
                

        # Create a DataFrame and save it as a CSV file
        df = pd.DataFrame(data)
        csv_file_path = os.path.join(save_pth, file + '.csv')
        # Split the DataFrame into training and validation sets
        # df = split_train_val(df)
        # Save the split DataFrame
        df.to_csv(csv_file_path, index=False)
        print(f"CSV file created: {csv_file_path}")

def make_csv_alive(pt_pth, targe_tsv,save_pth):
    # 定义可能的缺失值表示方式
    NA_VALUES = ['[Not Available]', '--', 'NA', 'N/A', '', 'Not Reported']

    files = os.listdir(pt_pth)
    # filer out non-file
    files = [file for file in files if os.path.isdir(os.path.join(pt_pth, file))]
    # open tsv file
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    tsv = pd.read_csv(targe_tsv, sep='\t')

    for file in files:
        pth = os.path.join(pt_pth, file, 'pt_files')
        if not os.path.exists(pth):
            continue
        pt_files = os.listdir(pth)
        data = []
        saved_list = []

        for pt_file in pt_files:
            pt_file_path = os.path.join(pth, pt_file)
            pt_file_path_ = pt_file.split('.pt')[0].split('_')[0]
            
            # Append the data
            # 检查pt_file_path_是否存在于saved_list中,如果不存在，则加入data
            if pt_file_path_ in saved_list:
                continue
            else:
            
                # Extract the slide name from the file name
                slide_name = pt_file.split('.')[-3].split('-')[0] + '-' + pt_file.split('.')[-3].split('-')[1] + '-' + pt_file.split('.')[-3].split('-')[2]
                
                tsv_row = tsv[tsv['cases.submitter_id'] == slide_name]
                if tsv_row.empty:
                    continue
                    
                target = tsv_row['project.project_id'].values[0] if not pd.isna(tsv_row['project.project_id'].values[0]) else np.nan
                vital_status = tsv_row['demographic.vital_status'].values[0] if not pd.isna(tsv_row['demographic.vital_status'].values[0]) else np.nan
                
                # 处理days_to_death和days_to_followup，确保能转换为float
                days_to_death = tsv_row['demographic.days_to_death'].values[0]
                if not pd.isna(days_to_death):
                    try:
                        days_to_death = float(days_to_death)
                    except:
                        days_to_death = np.nan
                
                days_to_followup = tsv_row['diagnoses.days_to_last_follow_up'].values[0]
                if not pd.isna(days_to_followup):
                    try:
                        days_to_followup = float(days_to_followup)
                    except:
                        days_to_followup = np.nan
                
                # 检查关键字段是否有有效值
                if pd.isna(target) or pd.isna(vital_status):
                    print(f"跳过病例 {slide_name}: 缺失分类标签或生存状态")
                    continue
                    
                if vital_status == 'Dead' and pd.isna(days_to_death):
                    print(f"跳过病例 {slide_name}: 死亡但缺失或无效的死亡时间")
                    continue
                    
                if vital_status != 'Dead' and pd.isna(days_to_followup):
                    print(f"跳过病例 {slide_name}: 存活但缺失或无效的随访时间")
                    continue

                if not pd.isna(target):
                    # target = target.values[0]
                    if target == 'TCGA-KIRP':
                        target = '0'
                    elif target == 'TCGA-KICH':
                        target = '1'
                    elif target == 'TCGA-PAAD':
                        target = '1'
                    elif target == 'TCGA-BRCA':
                        target = '1'
                    elif target == 'TCGA-COAD':
                        target = '1'
                    else:
                        raise ValueError(f"Unknown target: {target}")
                else:
                    target = 'unknown'
                
                # 事件标记: Dead=1, Alive或失访=0
                event = 1 if vital_status == 'Dead' else 0
                
                # 生存时间: 死亡时间 or 末次随访时间
                time = days_to_death if vital_status == 'Dead' else days_to_followup
                time = float(time)   # 将时间调整为天
                # 取两位小数
                time = round(time, 2) 
                
                saved_list.append(pt_file_path_)
                data.append({
                    'pt': pt_file_path_,
                    'target': target,
                    'mccvsplit': '0',  # Placeholder, adjust as needed
                    'slide_name': slide_name,
                    'event': event,
                    'time': time
                })

        # Create a DataFrame and save it as a CSV file
        df = pd.DataFrame(data)
        csv_file_path = os.path.join(save_pth, file + '_alive.csv')
        # Split the DataFrame into training and validation sets
        df = split_train_val(df)
        # Save the split DataFrame
        df.to_csv(csv_file_path, index=False)
        print(f"CSV file created: {csv_file_path}")


def split_train_val(df):
    # Split the DataFrame into training and validation sets
    df['mccvsplit'] = np.random.choice(['0', '1'], size=len(df), p=[0.8, 0.2])
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make CSV from PT files')
    parser.add_argument('--pt_pth', type=str, default=pt_pth, help='Path to the directory containing PT files')
    parser.add_argument('--tsv_pth', type=str, default=tsv_pth, help='Path to the TSV file containing targets')
    parser.add_argument('--save_pth', type=str, default=save_pth, help='Path to save the generated CSV files')
    
    args = parser.parse_args()
    
    # make_csv_alive(args.pt_pth, args.tsv_pth, args.save_pth)
    # make_csv_classify_COAD(args.pt_pth, args.tsv_pth, args.save_pth)
    make_csv_classify(args.pt_pth, args.tsv_pth, args.save_pth)
