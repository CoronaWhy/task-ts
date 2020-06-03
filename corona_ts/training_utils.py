def get_most_recent_file(file_path):
  list_of_files = glob.glob(file_path+"/*.pth") # * means all if need specific format then *.csv
  if len(list_of_files) > 1:
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file
  return None
 
 def make_wandb_config():
  pass
