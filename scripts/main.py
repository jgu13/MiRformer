from argument_parser import get_argument_parser
from mirLM import mirLM

def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    args_dict = vars(args)
        
    model = mirLM.create_model(**args_dict)
    model.run(model)

if __name__ == '__main__':
    main()
    