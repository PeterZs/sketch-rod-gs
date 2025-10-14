current_dir=$(pwd)
echo $current_dir

find_cmd="
    (
        find '$current_dir' -name '*.py' \
        -not -path '$current_dir/scene/*' \
        -not -path '$current_dir/utils/general_utils.py*' \
        -not -path '$current_dir/utils/sh_utils.py*' \
        -not -path '$current_dir/*/eigen-3.4.0/*' \
        -not -path '$current_dir/submodules/*' \
        -not -path '$current_dir/__pycache__/*' \
        -not -path '$current_dir/*/build/*' \
        -not -path '$current_dir/*/__pycache__/*' \
        -not -path '$current_dir/*/egg-info/*' \
        | tr '\n' ' '
    )
"
python_files=$(eval "$find_cmd")
echo "Processing: ""$python_files"

isort --settings-path $current_dir/formatter/pyproject.toml $python_files
black --config $current_dir/formatter/pyproject.toml $python_files
flake8 --config $current_dir/formatter/setup.cfg $python_files
# mypy $python_files
