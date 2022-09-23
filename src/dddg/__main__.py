import click

from dddg import loader
from dddg.solver import Solver


@click.command()
@click.argument('url')
def main(url: str):
    img = loader.load_image(url)
    solver = Solver(img)
    solver.run_inference()
    print(solver.solve_str())


if __name__ == '__main__':
    main()
