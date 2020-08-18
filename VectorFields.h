namespace SimulationUtilities{

	//need to be able to turn scalar field into rank 0 tensor field

	template<typename VectorType, size_t dimensions, size_t divisions, typename = std::enable_if_t<dimensions!=0 && divisions!=0>>
	class VectorField
	{
	public:
		static constexpr size_t dataSize = Template_Power<divisions, dimensions>::value;
	protected:
		typedef VectorField<VectorType, dimensions, divisions> SelfType;
		VectorType data[dataSize];
	public:
		VectorField()
		{
			for (size_t i = 0; i < dataSize; ++i)
			{
				data[i] = VectorType();
			}
		}
		VectorField(const std::vector<VectorType>& input)
		{
			std::copy(input.begin(), input.end(), data);
		}
		VectorField(VectorType* input)
		{
			std::copy(input, input + dataSize, data);
		}
		SelfType& operator+=(const SelfType& other)
		{
			for (size_t i = 0; i < dataSize; ++i)
			{
				data[i] += other.data[i];
			}
			return *this;
		}
		SelfType& operator-=(const SelfType& other)
		{
			for (size_t i = 0; i < dataSize; ++i)
			{
				data[i] -= other.data[i];
			}
			return *this;
		}
		SelfType& operator*=(double other){
			for (size_t i = 0; i < dataSize; ++i)
			{
				data[i] *= other;
			}
			return *this;
		}
		SelfType& operator*=(const VectorField<double, dimensions, divisions>& scalarField)
		{
			for (size_t i = 0; i < dataSize; ++i)
			{
				data[i] *= scalarField[i];
			}
			return *this;
		}
		SelfType& operator/=(double other){
			for (size_t i = 0; i < dataSize; ++i)
			{
				data[i] /= other;
			}
			return *this;
		}
		SelfType& operator/=(const VectorField<double, dimensions, divisions>& scalarField)
		{
			for (size_t i = 0; i < dataSize; ++i)
			{
				data[i] /= scalarField[i];
			}
			return *this;
		}

		VectorType& operator[](size_t index)
		{
			return data[index];
		}

		const VectorType& operator[](size_t index) const
		{
			return data[index];
		}

		const VectorType* begin() const
		{
			return data;
		}

		const VectorType* end() const
		{
			return data + dataSize;
		}

		TensorField<dimensions, 0, divisions, VectorType>& toTensor() const
		{
			return *(TensorField<dimensions, 0, divisions, VectorType>*)(this);
		}

		template<size_t rank, typename T, typename = std::enable_if_t<std::is_same<VectorType, Tensor<dimensions, rank, T>>::value>>
		operator TensorField<dimensions, rank, divisions, T>&()
		{
			return *(TensorField<dimensions, rank, divisions, T>*)(this);
		}
	};

	template<typename VectorType, size_t dimensions, size_t divisions>
	auto operator+(
		VectorField<VectorType, dimensions, divisions> left, const VectorField<VectorType, dimensions, divisions>& right)
	{
		return left += right;
	}
	template<typename VectorType, size_t dimensions, size_t divisions>
	auto operator-(
		VectorField<VectorType, dimensions, divisions> left, const VectorField<VectorType, dimensions, divisions>& right)
	{
		return left -= right;
	}
	template<typename VectorType, size_t dimensions, size_t divisions>
	auto operator*(VectorField<VectorType, dimensions, divisions> left, const double& right)
	{
		return left *= right;
	}
	template<typename VectorType, size_t dimensions, size_t divisions>
	auto operator*(const double& left, VectorField<VectorType, dimensions, divisions> right)
	{
		return right *= left;
	}
	template<typename VectorType, size_t dimensions, size_t divisions>
	auto operator*(
		VectorField<VectorType, dimensions, divisions> left, const VectorField<double, dimensions, divisions>& right)
	{
		return left *= right;
	}
	template<typename VectorType, size_t dimensions, size_t divisions, typename = std::enable_if_t<!std::is_same<double, VectorType>::value>>
	auto operator*(
		const VectorField<double, dimensions, divisions>& left, VectorField<VectorType, dimensions, divisions> right)
	{
		return right *= left;
	}
	template<typename VectorType, size_t dimensions, size_t divisions>
	auto operator/(VectorField<VectorType, dimensions, divisions> left, const double& right)
	{
		return left /= right;
	}
	template<typename VectorType, size_t dimensions, size_t divisions>
	auto operator/(VectorField<VectorType, dimensions, divisions> left, const VectorField<double, dimensions, divisions>& right)
	{
		return left /= right;
	}

	template<size_t dimensions, size_t divisions>
	using ScalarField = VectorField<double, dimensions, divisions>;

}