import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dims", help="list of dimensions", nargs='+')
    parser.add_argument("-a", "--alphas", help="list of alphas", nargs='+')
    parser.add_argument("-l", "--lambdas", help="list of lambdas", nargs='+')
    parser.add_argument("-e", "--experts", help="list of expert models", nargs='+')
    parser.add_argument("-m", "--antiexperts", help="list of antiexpert models", nargs='+')
    parser.add_argument("-o", "--output", help="output file")
    args = parser.parse_args()
    curr_dict = {}
    for i, dim in enumerate(args.dims):
        curr_dict[dim] = {}
        if args.alphas:
            curr_dict[dim]["alpha"] = float(args.alphas[i])
        if args.lambdas:
            curr_dict[dim]["lambda"] = float(args.lambdas[i])
        if args.experts:
            curr_dict[dim]["expert"] = args.experts[i]
        if args.antiexperts:
            curr_dict[dim]["antiexpert"] = args.antiexperts[i]
    
    with open(args.output, "w") as outfile: 
        json.dump(curr_dict, outfile)




    
if __name__ == '__main__':
    main()