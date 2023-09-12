import sys
import argparse
import healpix_sdpa_struct_loader

parser = argparse.ArgumentParser()
parser.add_argument("--healpix-res", "-hr", type=int, default=10, help="Resolution of the healpix for sampling.")
parser.add_argument("--patch-res", "-pr", type=int, default=8, help="Resolution of the healpix patches.")
parser.add_argument("--out-dir", "-o", type=str, default="../StructData", help="Directory to save structures.")
parser.add_argument("--use-4connectivity", action='store_true', help='use 4 neighboring for construction')
parser.add_argument("--use-euclidean", action='store_true', help='Use geodesic distance for weights')
parser.add_argument("--weight-type", '-w', type=str, default='identity', help="Weighting function on distances between nodes of the structure")
parser.add_argument("--num-hops", "-nh", type=int, default=1, help="Considered num of hops for each patch.")
parser.add_argument("--sdpa-normalization", '-sn', type=str, default="non", help="normalization method for sdpa convolutions")
parser.add_argument("--consider-outside-patch", action='store_true', help="if set, it means NOT to cut outside of patch")

args = parser.parse_args()

def main(argv):
    args = parser.parse_args(argv)

    print("=========== printing args ===========")
    for key, val in args.__dict__.items():
        print("{:_<40}: {}\n".format(key, val))  # check this for all kinds of formatting
    print("=" * 60 + "\n")

    struct_loader = healpix_sdpa_struct_loader.HealpixSdpaStructLoader(weight_type=args.weight_type,
                                                                       use_geodesic=not args.use_euclidean,
                                                                       use_4connectivity=args.use_4connectivity,
                                                                       normalization_method=args.sdpa_normalization,
                                                                       cutGraphForPatchOutside=not args.consider_outside_patch,
                                                                       load_save_folder=args.out_dir)
    n_patches, _ = struct_loader.getPatchesInfo(sampling_res=args.healpix_res, patch_res=args.patch_res)
    print("# patches={}".format(n_patches))
    # start_time_total = time.time()
    # for patch_id in range(n_patches):
    #     start_time_patch = time.time()
    #     # TODO: take a look at the generated neighbour structure
    #     struct_loader.getStruct(sampling_res=args.healpix_res, num_hops=args.num_hops, patch_res=args.patch_res, patch_id=patch_id)
    #     end_time_patch = time.time()
    #     print("Patch {} finished in {} seconds".format(patch_id, end_time_patch-start_time_patch), flush=True)
    # end_time_total = time.time()
    # print("--- {} seconds in total ---".format(end_time_total - start_time_total))

    struct_loader.getStruct(sampling_res=args.healpix_res, num_hops=args.num_hops, patch_res=None,
                            patch_id=None)


if __name__ == "__main__":
    main(sys.argv[1:])

