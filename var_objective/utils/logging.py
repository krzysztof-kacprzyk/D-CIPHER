import os
def create_output_dir(exp_name, exp_type):
    output_folder = os.path.join('experiments', 'results', exp_name, exp_type)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder

def create_logging_file_names(folder, dt):
    text_file = os.path.join(folder, f"{dt}.txt")
    csv_file = os.path.join(folder, f"{dt}_table.csv")
    meta_file = os.path.join(folder, f"{dt}_meta.p")

    return text_file, csv_file, meta_file 