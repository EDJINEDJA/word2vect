initialize_git :
	@echo "Git initialization"
	git init
	git add .
	git commit -m "My first commit"
	git branch -M origin main
	sleep 2
	git remote add origin https://github.com/EDJINEDJA/word2vect.git
	sleep 2
	git push -u origin main

pip_git:
	@echo "pushing ..."
	git add.
	git commit _m "first commit"
	git push -u origin main
env:
	@echo "setup env ..."
	pipenv install
activate:
	@echo "env activation ..."
	pipenv shell

setup : env
