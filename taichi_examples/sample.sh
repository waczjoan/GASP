E=1

data1="teddybear"
data2="fish_cup"
exp="${data1}_${data2}_clamp_exp"

in_dir1="output/${data1}/gs_flat/pseudomesh_info/ours_30000"
in_dir2="output/${data2}/gs_flat/pseudomesh_info/ours_30000"

out_dir1="simulations/3d/${data1}/${exp}"
out_dir2="simulations/3d/${data2}/${exp}"

s1="0.4 0.4 0.4"
s2="0.5 0.5 0.5"

o1="0.30 0.5 0.30"
o2="0.25 0.0 0.25"

python -u ./taichi_examples/demo/teddybear_carpet.py \
  -i $in_dir1 \
     $in_dir2 \
  -o $out_dir1 \
     $out_dir2 \
  --scales $s1 \
           $s2 \
  --offsets $o1 \
            $o2 \
  --E $E --skip 1 --material sand

rm -rf $out_dir1/objects $out_dir2/objects

python -u render_multiple_gaussians.py \
   --model_paths \
       $in_dir1/../.. \
       $in_dir2/../.. \
   --sym_dirnames \
       $out_dir1 \
       $out_dir2
