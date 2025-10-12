for arg in "$@"
do
    echo $arg
    isort $arg
    black $arg
    flake8 $arg
    # mypy $arg
done
