import os, sys, shutil
FHMCLIB = "/home/nam4/"
sys.path.append(FHMCLIB)
import FHMCAnalysis
import FHMCAnalysis.moments.win_patch.windows as win
import FHMCSimulation.helper.window_helper as hP

# Overwrite existing inputs
overwrite = True

# Establish bounds for windows
ntot_max = 600
final_window_width = 20
num_windows = 24
num_overlap = 6

# Need installation information
install_dir = "/home/nam4/FHMCSimulation/"
binary = install_dir+"/bin/fhmc_tmmc"
git_head = install_dir+"/.git/logs/HEAD"
jobs_per = 12
scratch_dir = "/scratch/nam4/"
q = ""
tag = "lam1.5(2)"
hours = 72
	
# Window settings
input_name = "input.json"
prefix = "./"
beta = 1.0/0.90
bounds = win.ntot_window_scaling (ntot_max, final_window_width, num_windows, num_overlap)

sett = {}
sett["beta"] = beta
for w in range(num_windows):
	dname = prefix+"/"+str(w+1)
	if ((str(w+1) in os.listdir(prefix)) and overwrite):
		shutil.rmtree(dname, pure_settings)
	os.makedirs(dname)

	sett["bounds"] = bounds[w]
	hP.make_input (dname+"/"+input_name, sett, hP.pure_settings)
	hP.make_sleeper (dname+"/sleeper.sh")

hP.raritan_sbatch (num_windows, binary, git_head, tag, prefix, input_name, jobs_per, q, hours, scratch_dir)
