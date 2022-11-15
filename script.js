let list = document.querySelectorAll(".collapse");

list.forEach((desc) => {
	desc.addEventListener("click", () => {
		desc.classList.toggle("open");
	});
});
