initialize_git :
	@echo "Git initialization"
	git init
	git add .
	git commit -m "My first commit"
	git branch -M origin main
	git remote add origin https://github.com/EDJINEDJA/word2vect.git
	git push -u origin main
pip_git:
	@echo "pushing ..."
	git add .
	git commit -m $(COMMIT)
	git push -u origin master
pull_git:
	@echo "pulling ..."
	git pull origin master
env:
	@echo "setup env ..."
	pipenv install
activate:
	@echo "env activation ..."
	pipenv shell
test:
	python app.py

setup : env
