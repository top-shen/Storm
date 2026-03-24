workdir = "workdir"
tag = "processd_day_dj30_17"
batch_size = 5

processor = dict(
    type = "Processor",
    path_params = {
        "prices": [
            {
                "type": "fmp",
                "path": "workdir/dj30/price",
            }
        ],
    },
    start_date = "1994-03-01",
    end_date = "2024-07-01",
    interval = "1d",
    assets_path = "configs/_asset_list_/dj30_17.json",
    workdir = workdir,
    tag = tag,
)
